#!/usr/bin/env python
# coding: utf-8

"""
RECOVERY SCRIPT
===============
Este script repara la ejecución incompleta de cluster_embeddings.py.
Toma los labels/centroides ya generados, evalúa cuál es el mejor
y genera la carpeta 'final' con A_ideal_best.parquet.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

# -------------------------------------------------------
# CONFIG (Mismas rutas que tu script original)
# -------------------------------------------------------
CLUSTERS_DIR = "/lustre/home/dante/compartido/clusters/"
K_LIST = [8, 12, 16, 20, 24, 32, 40]
N_BOOTSTRAPS = 20

# -------------------------------------------------------
# LOAD EMBEDDINGS (Necesarios para calcular Silhouette)
# -------------------------------------------------------
print("[INFO] Loading pre-calculated embeddings...")
umap_path = os.path.join(CLUSTERS_DIR, "UMAP_embeddings.npy")
pca_path = os.path.join(CLUSTERS_DIR, "PCA_embeddings.npy")

if os.path.exists(umap_path):
    print("   -> Found UMAP embeddings.")
    Xembed = np.load(umap_path)
elif os.path.exists(pca_path):
    print("   -> Found PCA embeddings (UMAP missing).")
    Xembed = np.load(pca_path)
else:
    raise FileNotFoundError("No se encontraron embeddings (UMAP o PCA) en la carpeta clusters.")

# -------------------------------------------------------
# HELPER: BOOTSTRAP STABILITY
# -------------------------------------------------------
def bootstrap_ari(labels):
    # Si hay ruido (-1), lo ignoramos para estabilidad o lo tratamos como cluster
    valid_mask = labels != -1
    if valid_mask.sum() < len(labels) * 0.5:
        return 0.0 # Demasiado ruido
    
    ari_scores = []
    n = len(labels)
    # Hacemos menos bootstraps para que el recovery sea rápido
    for _ in range(10): 
        idx = np.random.choice(n, n, replace=True)
        ari = adjusted_rand_score(labels[idx], labels)
        ari_scores.append(ari)
    return float(np.mean(ari_scores))

# -------------------------------------------------------
# RE-EVALUATE SAVED RESULTS
# -------------------------------------------------------
results = []

print("[INFO] Re-evaluating existing clusters...")

# 1. Check KMeans
for K in K_LIST:
    path = os.path.join(CLUSTERS_DIR, f"KMeans/K{K}")
    if os.path.exists(path):
        try:
            labels = np.load(os.path.join(path, "labels.npy"))
            sil = silhouette_score(Xembed, labels)
            stab = bootstrap_ari(labels)
            results.append({"method": "kmeans", "K": K, "score": sil + stab*2.0})
            print(f"   -> KMeans K={K}: Score={sil + stab*2.0:.4f}")
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
            results.append({"method": "gmm", "K": K, "score": sil + stab*2.0})
            print(f"   -> GMM K={K}: Score={sil + stab*2.0:.4f}")
        except Exception as e:
            print(f"   [WARN] Could not load GMM K={K}: {e}")

# 3. Check HDBSCAN
h_path = os.path.join(CLUSTERS_DIR, "HDBSCAN")
if os.path.exists(h_path):
    try:
        labels = np.load(os.path.join(h_path, "labels.npy"))
        # Filtrar ruido (-1) para silhouette
        valid = labels != -1
        if valid.sum() > 100:
            sil = silhouette_score(Xembed[valid], labels[valid])
            stab = bootstrap_ari(labels)
            k_found = len(np.unique(labels[valid]))
            results.append({"method": "hdbscan", "K": k_found, "score": sil + stab*2.0})
            print(f"   -> HDBSCAN K={k_found}: Score={sil + stab*2.0:.4f}")
    except Exception as e:
        print(f"   [WARN] Could not load HDBSCAN: {e}")

# -------------------------------------------------------
# SELECT BEST & EXPORT
# -------------------------------------------------------
if not results:
    raise RuntimeError("No se pudieron cargar resultados previos. ¿Están vacías las carpetas?")

dfres = pd.DataFrame(results)
best = dfres.sort_values("score", ascending=False).iloc[0]

print("\n=== WINNER CONFIG ===")
print(best)

best_method = best["method"]
best_K = int(best["K"])

# Cargar centroides del ganador
centroids = None

if best_method == "kmeans":
    centroids = np.load(os.path.join(CLUSTERS_DIR, f"KMeans/K{best_K}/centroids.npy"))
elif best_method == "gmm":
    centroids = np.load(os.path.join(CLUSTERS_DIR, f"GMM/K{best_K}/means.npy"))
elif best_method == "hdbscan":
    labels = np.load(os.path.join(CLUSTERS_DIR, "HDBSCAN/labels.npy"))
    # Recalcular centroides para HDBSCAN
    unique_labels = [c for c in np.unique(labels) if c >= 0]
    centroids = np.vstack([Xembed[labels == cid].mean(axis=0) for cid in unique_labels])

# Guardar
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

print("\n✅ RECOVERY COMPLETE.")
print(f"A_ideal generated at: {out_path}")