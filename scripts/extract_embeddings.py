#!/usr/bin/env python
# coding: utf-8

"""
Extrae embeddings de jugadas usando el backbone SSL entrenado.
Requiere:
 - train_ssl.py (por el modelo DynamicEncoder)
 - dataset_dynamic.py (por dataloader y build_graphs_from_batch)
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# 🔧 Configuración
BACKBONE_PATH = "/lustre/home/dante/compartido/models/backbone_ssl.pth.ckpt_epoch50"
OUTPUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

# Asegurar carpeta
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cargar clases desde train_ssl.py
from train_ssl import DynamicEncoder, HIDDEN_DIM, build_graphs_from_batch, dataloader, device

# 🔹 Cargar modelo
print(f"[INFO] Cargando backbone desde {BACKBONE_PATH} ...")
encoder = DynamicEncoder(in_dim=6, hidden_dim=HIDDEN_DIM).to(device)
state = torch.load(BACKBONE_PATH, map_location=device)
encoder.load_state_dict(state)
encoder.eval()
print("[INFO] Backbone cargado correctamente en modo eval.")

# 🔹 Extraer embeddings por jugada
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

# 🔹 Guardar resultados
import numpy as np

embeddings = np.stack(embeddings)
df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
df["play_id"] = play_ids

df.to_parquet(OUTPUT_PATH, index=False)
print(f"[✅] Embeddings guardados en: {OUTPUT_PATH}")
print(f"[INFO] Total de jugadas procesadas: {len(df)}")
