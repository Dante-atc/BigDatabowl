#!/usr/bin/env python
# coding: utf-8

"""
Cluster Recovery & Selection Utility
====================================

This script is a fail-safe utility designed to repair or finalize the clustering
process if the main pipeline was interrupted. It scans the output directories
for generated labels and centroids, re-evaluates them using Silhouette and 
Stability metrics, and exports the optimal configuration as 'A_ideal'.

Workflow:
1. Load pre-computed embeddings (UMAP/PCA).
2. Iterate through existing cached results (KMeans, GMM, HDBSCAN).
3. Compute validation metrics on the fly.
4. Select the winner and publish to 'clusters/final'.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
CLUSTERS_DIR = "/lustre/home/dante/compartido/clusters/"
K_LIST = [8, 12, 16, 20, 24, 32, 40]

# -------------------------------------------------------
# LOAD EMBEDDINGS
# -------------------------------------------------------
print("[INFO] Loading pre-calculated embeddings for validation...")
umap_path = os.path.join(CLUSTERS_DIR, "UMAP_embeddings.npy")
pca_path = os.path.join(CLUSTERS_DIR, "PCA_embeddings.npy")

if os.path.exists(umap_path):
    print("   -> Found UMAP embeddings.")
    Xembed = np.load(umap_path)
elif os.path.exists(pca_path):
    print("   -> Found PCA embeddings (UMAP missing, using PCA).")
    Xembed = np.load(pca_path)
else:
    raise FileNotFoundError("Critical: No embeddings found in clusters folder.")

# -------------------------------------------------------
# HELPER: BOOTSTRAP STABILITY
# -------------------------------------------------------
def bootstrap_ari(labels):
    """
    Computes stability via Adjusted Rand Index on resampled data.
    """
    valid_mask = labels != -1
    if valid_mask.sum() < len(labels) * 0.5:
        return 0.0  # Too much noise
    
    ari_scores = []
    n = len(labels)
    # Perform fewer bootstraps for quick recovery
    for _ in range(10): 
        idx = np.random.choice(n, n, replace=True)
        ari = adjusted_rand_score(labels[idx], labels)
        ari_scores.append(ari)
    return float(np.mean(ari_scores))

# -------------------------------------------------------
# RE-EVALUATION LOOP
# -------------------------------------------------------
results = []

print("[INFO] Re-evaluating existing cluster artifacts...")

# 1. Check KMeans
for K in K_LIST:
    path = os.path.join(CLUSTERS_DIR, f"KMeans/K{K}")
    if os.path.exists(path):
        try:
            labels = np.load(os.path.join(path, "labels.npy"))
            sil = silhouette_score(Xembed, labels)
            stab = bootstrap_ari(labels)
            score = sil + stab * 2.0
            results.append({"method": "kmeans", "K": K, "score": score})
            print(f"   -> KMeans K={K}: Score={score:.4f}")
        except Exception as e:
            print(f"   [WARN] Could not load KMeans K={K}: {e}")

# 2. Check GMM
for K in K_LIST:
    path = os.path.join(CLUSTERS_DIR, f"GMM/K{K}")
    if os.path.exists(path):
        try:
            labels = np.load(os.path.join(path, "labels.npy"))
            sil = silhouette_score(Xembed, labels)
            stab = bootstrap_ari(labels)
            score = sil + stab * 2.0
            results.append({"method": "gmm", "K": K, "score": score})
            print(f"   -> GMM K={K}: Score={score:.4f}")
        except Exception as e:
            print(f"   [WARN] Could not load GMM K={K}: {e}")

# 3. Check HDBSCAN
h_path = os.path.join(CLUSTERS_DIR, "HDBSCAN")
if os.path.exists(h_path):
    try:
        labels = np.load(os.path.join(h_path, "labels.npy"))
        valid = labels != -1
        if valid.sum() > 100:
            sil = silhouette_score(Xembed[valid], labels[valid])
            stab = bootstrap_ari(labels)
            k_found = len(np.unique(labels[valid]))
            score = sil + stab * 2.0
            results.append({"method": "hdbscan", "K": k_found, "score": score})
            print(f"   -> HDBSCAN K={k_found}: Score={score:.4f}")
    except Exception as e:
        print(f"   [WARN] Could not load HDBSCAN: {e}")

# -------------------------------------------------------
# SELECTION & EXPORT
# -------------------------------------------------------
if not results:
    raise RuntimeError("Could not load previous results. Are the output folders empty?")

dfres = pd.DataFrame(results)
best = dfres.sort_values("score", ascending=False).iloc[0]

print("\n=== OPTIMAL CONFIGURATION ===")
print(best)

best_method = best["method"]
best_K = int(best["K"])

# Retrieve best centroids
centroids = None

if best_method == "kmeans":
    centroids = np.load(os.path.join(CLUSTERS_DIR, f"KMeans/K{best_K}/centroids.npy"))
elif best_method == "gmm":
    centroids = np.load(os.path.join(CLUSTERS_DIR, f"GMM/K{best_K}/means.npy"))
elif best_method == "hdbscan":
    labels = np.load(os.path.join(CLUSTERS_DIR, "HDBSCAN/labels.npy"))
    unique_labels = [c for c in np.unique(labels) if c >= 0]
    centroids = np.vstack([Xembed[labels == cid].mean(axis=0) for cid in unique_labels])

# Prepare DataFrame
df_cent = pd.DataFrame(centroids, columns=[f"dim_{i}" for i in range(centroids.shape[1])])
df_cent["cluster_id"] = range(len(df_cent))
df_cent["method"] = best_method
df_cent["K"] = best_K

final_dir = os.path.join(CLUSTERS_DIR, "final")
os.makedirs(final_dir, exist_ok=True)

out_path = os.path.join(final_dir, "A_ideal_best.parquet")
df_cent.to_parquet(out_path, index=False)

with open(os.path.join(final_dir, "metadata_recovered.json"), "w") as f:
    json.dump(best.to_dict(), f, indent=4)

print("\n[SUCCESS] Recovery Complete.")
print(f"A_ideal template generated at: {out_path}")