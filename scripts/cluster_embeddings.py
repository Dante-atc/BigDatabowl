##!/usr/bin/env python
# coding: utf-8

"""
Cluster Ensemble para Embeddings — Versión Máxima Calidad
Genera plantillas A_ideal usando:
 - PCA
 - UMAP
 - KMeans (muchos K)
 - Gaussian Mixture Models
 - HDBSCAN
 - Bootstrap Stability (ARI)
 - Selección automática del mejor método + K
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

#############################################
# CONFIG
#############################################

INPUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"
OUT_DIR = "/lustre/home/dante/compartido/clusters/"
os.makedirs(OUT_DIR, exist_ok=True)

K_LIST = [8, 12, 16, 20, 24, 32, 40]
N_BOOTSTRAPS = 20

#############################################
# LOAD DATA
#############################################

print("[INFO] Loading embeddings...")
df = pd.read_parquet(INPUT_PATH)
embed_cols = [c for c in df.columns if "dim_" in c]
X = df[embed_cols].values

#############################################
# NORMALIZE
#############################################

print("[INFO] Normalizing...")
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

#############################################
# PCA
#############################################

print("[INFO] PCA 128 → 32...")
pca = PCA(n_components=32)
Xpca = pca.fit_transform(Xn)
np.save(os.path.join(OUT_DIR, "PCA_embeddings.npy"), Xpca)

#############################################
# UMAP 
#############################################

try:
    import umap
    print("[INFO] UMAP 32 → 16...")
    reducer = umap.UMAP(n_components=16, n_neighbors=50, min_dist=0.1)
    Xumap = reducer.fit_transform(Xpca)
    np.save(os.path.join(OUT_DIR, "UMAP_embeddings.npy"), Xumap)
    Xembed = Xumap
except:
    print("[WARN] UMAP not available, using PCA 32 dims")
    Xembed = Xpca

#############################################
# CLUSTERING
#############################################

def run_kmeans(K):
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(Xembed)
    return labels, km.cluster_centers_

def run_gmm(K):
    g = GaussianMixture(n_components=K, covariance_type="full", n_init=10, random_state=42)
    labels = g.fit_predict(Xembed)
    return labels, g.means_

def run_hdbscan():
    h = hdbscan.HDBSCAN(min_cluster_size=50, cluster_selection_epsilon=0.5)
    labels = h.fit_predict(Xembed)
    probs = h.probabilities_
    return labels, probs

#############################################
# BOOTSTRAP STABILITY
#############################################

def bootstrap_ari(labels):
    ari_scores = []
    n = len(labels)
    for _ in range(N_BOOTSTRAPS):
        idx = np.random.choice(n, n, replace=True)
        ari = adjusted_rand_score(labels[idx], labels)
        ari_scores.append(ari)
    return float(np.mean(ari_scores))

#############################################
# EXECUTION
#############################################

results = []

print("[INFO] Executing K-Means + GMM for multiple Ks...")
for K in K_LIST:
    km_labels, km_centroids = run_kmeans(K)
    gmm_labels, gmm_centroids = run_gmm(K)

    silhouette_km = silhouette_score(Xembed, km_labels)
    silhouette_gmm = silhouette_score(Xembed, gmm_labels)

    stability_km = bootstrap_ari(km_labels)
    stability_gmm = bootstrap_ari(gmm_labels)

    out_km = f"KMeans/K{K}"
    out_gmm = f"GMM/K{K}"
    os.makedirs(os.path.join(OUT_DIR, out_km), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, out_gmm), exist_ok=True)

    np.save(os.path.join(OUT_DIR, out_km, "labels.npy"), km_labels)
    np.save(os.path.join(OUT_DIR, out_km, "centroids.npy"), km_centroids)

    np.save(os.path.join(OUT_DIR, out_gmm, "labels.npy"), gmm_labels)
    np.save(os.path.join(OUT_DIR, out_gmm, "means.npy"), gmm_centroids)

    results.append({
        "method": "kmeans",
        "K": K,
        "silhouette": silhouette_km,
        "stability": stability_km
    })

    results.append({
        "method": "gmm",
        "K": K,
        "silhouette": silhouette_gmm,
        "stability": stability_gmm
    })

print("[INFO] Executing HDBSCAN...")
h_labels, h_probs = run_hdbscan()
os.makedirs(os.path.join(OUT_DIR, "HDBSCAN"), exist_ok=True)
np.save(os.path.join(OUT_DIR, "HDBSCAN/labels.npy"), h_labels)
np.save(os.path.join(OUT_DIR, "HDBSCAN/probs.npy"), h_probs)


valid_mask = h_labels >= 0
num_valid = np.sum(valid_mask)
num_clusters_h = len(np.unique(h_labels[valid_mask]))

if num_valid > 100 and num_clusters_h > 1:
    
    sil_h = silhouette_score(Xembed[valid_mask], h_labels[valid_mask])
    stab_h = bootstrap_ari(h_labels[valid_mask])
    
    results.append({
        "method": "hdbscan",
        "K": num_clusters_h,
        "silhouette": sil_h,
        "stability": stab_h
    })
else:
    print(f"[WARN] HDBSCAN found only 1 cluster. Ignoring.")
    

#############################################
# BEST METHOD SELECTION
#############################################

dfres = pd.DataFrame(results)
dfres["score"] = dfres["silhouette"] + dfres["stability"] * 2.0

best = dfres.sort_values("score", ascending=False).iloc[0]
print("\n=== BEST CONFIG ===")
print(best)

#############################################
# EXPORT FINAL A_ideal
#############################################

best_method = best["method"]
best_K = int(best["K"])

if best_method == "kmeans":
    centroids = np.load(os.path.join(OUT_DIR, f"KMeans/K{best_K}/centroids.npy"))
elif best_method == "gmm":
    centroids = np.load(os.path.join(OUT_DIR, f"GMM/K{best_K}/means.npy"))
else:
    labels = np.load(os.path.join(OUT_DIR, "HDBSCAN/labels.npy"))
    centroids = np.vstack([
        Xembed[labels == cid].mean(axis=0)
        for cid in np.unique(labels) if cid >= 0
    ])

df_cent = pd.DataFrame(centroids, columns=[f"dim_{i}" for i in range(centroids.shape[1])])
df_cent["cluster_id"] = range(len(df_cent))
df_cent["method"] = best_method
df_cent["K"] = best_K

final_dir = os.path.join(OUT_DIR, "final")
os.makedirs(final_dir, exist_ok=True)

out_path = os.path.join(final_dir, "A_ideal_best.parquet")
df_cent.to_parquet(out_path, index=False)

with open(os.path.join(final_dir, "metadata.json"), "w") as f:
    json.dump(best.to_dict(), f, indent=4)

print("\nA_ideal templates saved in:")
print(out_path)
