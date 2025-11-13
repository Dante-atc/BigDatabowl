#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[3]:


import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


# # Config

# In[ ]:


PROCESSED_PATH = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"

# Parámetros generales
BATCH_SIZE = 8
NUM_WORKERS = 4

#Semillas fijas
torch.manual_seed(42)
np.random.seed(42)


# # Dataset

# In[5]:


class DynamicPlayDataset(Dataset):
    """
    Dataset que entrega pares (X_t, X_t+1) dinámicamente para entrenamiento SSL.
    X_t y X_t+1 representan el estado del campo en frames consecutivos.
    """
    def __init__(self, parquet_path, seq_len=2):
        super().__init__()
        self.seq_len = seq_len
        print(f"Cargando dataset procesado desde {parquet_path}...", flush=True)
        self.data = pd.read_parquet(parquet_path)

        # Validación básica
        required_cols = ["game_id", "play_id", "frame_id", "nfl_id", "x", "y", "s", "a", "o", "dir"]
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el parquet: {missing}")

        # Agrupamos por jugada
        self.plays = []
        for (gid, pid), df in tqdm(self.data.groupby(["game_id", "play_id"]), desc="Agrupando jugadas"):
            frames = sorted(df["frame_id"].unique())
            self.plays.append({
                "game_id": gid,
                "play_id": pid,
                "frames": frames,
                "df": df
            })

        print(f"✅ Total de jugadas cargadas: {len(self.plays)}")

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        play = self.plays[idx]
        df = play["df"]
        frames = play["frames"]

        if len(frames) < self.seq_len:
            # Si la jugada es demasiado corta, se descarta
            return None, None

        # Selecciona un frame al azar dentro de la jugada
        t = np.random.randint(0, len(frames) - 1)
        f_t, f_tp1 = frames[t], frames[t + 1]

        feat_cols = ["x", "y", "s", "a", "o", "dir"]

        # Asegurar consistencia de IDs
        nfl_ids = sorted(df["nfl_id"].unique())
        id_map = {nid: i for i, nid in enumerate(nfl_ids)}

        # Filtrar por frame
        df_t = df[df["frame_id"] == f_t].copy()
        df_tp1 = df[df["frame_id"] == f_tp1].copy()

        # Rellenar jugadores faltantes con NaN → 0
        def fill_players(df_frame):
            mat = np.zeros((len(nfl_ids), len(feat_cols)), dtype=np.float32)
            for _, row in df_frame.iterrows():
                i = id_map[row["nfl_id"]]
                mat[i] = row[feat_cols].values
            return mat

        X_t = fill_players(df_t)
        X_tp1 = fill_players(df_tp1)

        return torch.tensor(X_t), torch.tensor(X_tp1)


# # Collate data

# In[6]:


def dynamic_collate(batch):
    """
    Permite batching de jugadas con distinto número de jugadores.
    Si una jugada es inválida (None), la descarta del batch.
    """
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


# # Dataloader

# In[7]:


dataset = DynamicPlayDataset(PROCESSED_PATH)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=dynamic_collate
)


# # Test

# In[8]:


for batch in dataloader:
    if batch[0] is None:
        continue
    X_t, X_tp1 = batch
    print(f"Lote con forma: X_t={X_t.shape}, X_tp1={X_tp1.shape}")
    break

print("✅ Dataloader dinámico listo para entrenamiento SSL.")


# GRAPH BUILDER for Dynamic R-GCN

# In[ ]:


from torch_geometric.data import Data

def build_graphs_from_batch(X_batch, max_distance=10.0):
    """
    Convierte un batch tensorial [B, N, F] → lista de grafos PyG (Data objects)
    Cada grafo representa una jugada (frame) donde los nodos son jugadores
    y las aristas se definen según distancia espacial.

    Parámetros
    ----------
    X_batch : torch.Tensor
        Tensor de forma [B, N, F] del dataloader (jugadas por batch)
    max_distance : float
        Distancia máxima (yardas) para crear una arista entre jugadores.

    Retorna
    -------
    graphs : list[torch_geometric.data.Data]
        Lista de grafos por jugada.
    """
    graphs = []

    if X_batch is None:
        return graphs

    B, N, F = X_batch.shape

    for b in range(B):
        x = X_batch[b]  # [N, F]
        # Solo consideramos nodos válidos (jugadores con features no todos ceros)
        valid_mask = (x.abs().sum(dim=1) > 0)
        x_valid = x[valid_mask]
        num_nodes = x_valid.size(0)

        if num_nodes <= 1:
            continue  # Jugada inválida o sin datos útiles

        # Calculamos matriz de distancias
        pos = x_valid[:, :2]  # columnas x, y
        dist = torch.cdist(pos, pos, p=2)

        # Creamos aristas donde distancia < max_distance y no sea self-loop
        edge_index = (dist < max_distance).nonzero(as_tuple=False).T
        # eliminamos self-loops (i == j)
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

        # Asignamos edge_type (por ahora sin distinción ofensiva/defensiva)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        data = Data(
            x=x_valid,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes
        )

        graphs.append(data)

    return graphs

