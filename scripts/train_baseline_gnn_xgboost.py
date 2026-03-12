#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASELINE: Simple GNN + XGBoost
===============================
This script trains a simple (non-SSL) GNN baseline to compare against the
full SSL + R-GCN pipeline. The goal is to demonstrate the added value of:
  1. Self-supervised pretraining (contrastive + reconstruction)
  2. Relational graph convolution (R-GCN with edge types)
  3. Temporal modeling (GRU cell)

Baseline Architecture:
  - Simple 2-layer GCN (no relation types, no SSL pretraining)
  - Mean-pooled play-level embeddings
  - XGBoost classifier for defensive success prediction

Pipeline:
  Phase A: Train simple GCN encoder (supervised, end-to-end)
  Phase B: Extract play-level embeddings
  Phase C: Train XGBoost on embeddings + context features
  Phase D: Compute baseline DCI/DIS and compare with SSL version

Designed to run on the YUCA HPC cluster.

Inputs:
  - /lustre/proyectos/p037/datasets/processed/plays_processed.parquet
  - supplementary_data.csv

Outputs:
  - baseline_gnn_xgboost_metrics.parquet
  - baseline_comparison_results.csv
"""

import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Will use HistGradientBoosting as fallback.")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# PyG
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[ERROR] torch_geometric not installed. Cannot run GNN baseline.")
    sys.exit(1)

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Paths (YUCA HPC)
BASE_DIR = "/lustre/home/dante/compartido"
RAW_DATA = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"
RAW_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"
SUPP_PATH = os.path.join(RAW_DIR, "supplementary_data.csv")

OUT_DIR = os.path.join(BASE_DIR, "baselines")
os.makedirs(OUT_DIR, exist_ok=True)

# Model hyperparameters
IN_DIM = 6          # x, y, s, a, o, dir
HIDDEN_DIM = 128    # Deliberately smaller than SSL model (512)
EMBED_DIM = 64      # Final embedding dimension
EPOCHS = 50         # Fewer epochs (no SSL pretraining needed)
BATCH_SIZE = 64
LR = 1e-3
EDGE_THRESHOLD = 10.0  # yards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")


# -------------------------------------------------------
# SIMPLE GCN ENCODER (Baseline — no R-GCN, no GRU)
# -------------------------------------------------------
class SimpleGCNEncoder(nn.Module):
    """
    Simple 2-layer GCN for comparison with the SSL R-GCN + Transformer + GRU.
    Key differences from the full model:
      - Uses GCNConv instead of RGCNConv (ignores edge types)
      - No TransformerConv layer
      - No GRU temporal module
      - Smaller hidden dimension (128 vs 512)
    """
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        # Mean pool across nodes to get play-level embedding
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        h = self.fc(h)
        return h


class SupervisedGNNModel(nn.Module):
    """End-to-end supervised GNN: encoder -> classifier."""
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.encoder = SimpleGCNEncoder(in_dim, hidden_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, batch=None):
        embedding = self.encoder(x, edge_index, batch)
        logit = self.classifier(embedding).squeeze(-1)
        return logit, embedding


# -------------------------------------------------------
# DATASET: Loads plays and builds graphs on-the-fly
# -------------------------------------------------------
class PlayGraphDataset(Dataset):
    """Loads processed plays and converts to PyG graphs with labels."""

    def __init__(self, play_data, labels, edge_threshold=10.0):
        self.plays = play_data
        self.labels = labels
        self.edge_threshold = edge_threshold

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        feat_matrix = self.plays[idx]  # [N, 6]
        label = self.labels[idx]

        x = torch.tensor(feat_matrix, dtype=torch.float32)

        # Remove zero-padded players
        valid_mask = x.abs().sum(dim=1) > 0
        x = x[valid_mask]
        num_nodes = x.size(0)

        if num_nodes <= 1:
            # Cannot build meaningful graph
            x = torch.zeros((2, 6))
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        else:
            # Build edges by distance threshold
            pos = x[:, :2]
            dist = torch.cdist(pos, pos, p=2)
            edge_index = (dist < self.edge_threshold).nonzero(as_tuple=False).T
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float32))
        return data


# -------------------------------------------------------
# PHASE A: DATA LOADING & GRAPH CONSTRUCTION
# -------------------------------------------------------
def load_and_prepare_data():
    """Load raw plays, merge with labels, prepare graph data."""
    print("[PHASE A] Loading data...")

    # Load tracking data
    print(f"  Loading: {RAW_DATA}")
    df = pd.read_parquet(RAW_DATA)

    # Load supplementary data
    print(f"  Loading: {SUPP_PATH}")
    supp = pd.read_csv(SUPP_PATH, low_memory=False)

    # Standardize column names
    cols_map = {
        "gameId": "game_id", "playId": "play_id",
        "passResult": "pass_result", "expectedPointsAdded": "epa",
        "expected_points_added": "epa",
        "yardsToGo": "yards_to_go", "defendersInTheBox": "defenders_in_the_box",
    }
    supp.rename(columns=cols_map, inplace=True)

    # Filter to pass plays
    valid_pass = ["C", "I", "S", "IN"]
    supp_pass = supp[supp["pass_result"].isin(valid_pass)].copy()
    supp_pass["defensive_success"] = (supp_pass["epa"] <= 0).astype(int)

    # Group tracking data by play → feature matrices
    feat_cols = ["x", "y", "s", "a", "o", "dir"]
    play_matrices = {}

    print("  Grouping plays into feature matrices...")
    for (gid, pid), group in df.groupby(["game_id", "play_id"]):
        # Use middle frame (snap + few frames)
        frames = sorted(group["frame_id"].unique())
        mid_frame = frames[len(frames) // 2]
        frame_data = group[group["frame_id"] == mid_frame]

        mat = frame_data[feat_cols].values.astype(np.float32)
        mat = np.nan_to_num(mat)
        play_matrices[(gid, pid)] = mat

    # Match with labels
    plays_list = []
    labels_list = []
    meta_list = []  # For later context feature extraction

    for _, row in supp_pass.iterrows():
        key = (row["game_id"], row["play_id"])
        if key in play_matrices:
            plays_list.append(play_matrices[key])
            labels_list.append(row["defensive_success"])
            meta_list.append({
                "game_id": row["game_id"],
                "play_id": row["play_id"],
                "epa": row["epa"],
                "pass_result": row["pass_result"],
                "down": row.get("down", 1),
                "yards_to_go": row.get("yards_to_go", 10),
                "defenders_in_the_box": row.get("defenders_in_the_box", 6),
            })

    print(f"  Matched {len(plays_list)} plays with labels.")
    return plays_list, labels_list, meta_list


# -------------------------------------------------------
# PHASE B: TRAIN SIMPLE GCN (Supervised, End-to-End)
# -------------------------------------------------------
def train_supervised_gnn(plays_list, labels_list):
    """Train simple GCN encoder end-to-end for defensive success."""
    print("\n[PHASE B] Training Simple GCN (supervised)...")

    dataset = PlayGraphDataset(plays_list, labels_list, EDGE_THRESHOLD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda batch: Batch.from_data_list(batch))

    model = SupervisedGNNModel(IN_DIM, HIDDEN_DIM, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.squeeze())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    return model


# -------------------------------------------------------
# PHASE C: EXTRACT EMBEDDINGS + TRAIN XGBOOST
# -------------------------------------------------------
def extract_embeddings_and_train_xgb(model, plays_list, labels_list, meta_list):
    """Extract GCN embeddings, combine with context, train XGBoost."""
    print("\n[PHASE C] Extracting embeddings & training XGBoost...")

    dataset = PlayGraphDataset(plays_list, labels_list, EDGE_THRESHOLD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda batch: Batch.from_data_list(batch))

    model.eval()
    all_embeddings = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, embeddings = model(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(embeddings.cpu().numpy())
            all_logits.append(torch.sigmoid(logits).cpu().numpy())

    embeddings_np = np.vstack(all_embeddings)
    gnn_probs = np.concatenate(all_logits)

    print(f"  Embeddings shape: {embeddings_np.shape}")

    # Build context features
    context_features = []
    for m in meta_list:
        context_features.append([
            float(m.get("down", 1)),
            float(m.get("yards_to_go", 10)),
            float(m.get("defenders_in_the_box", 6)),
        ])
    context_np = np.array(context_features, dtype=np.float32)

    # Combine embeddings + context
    X_combined = np.hstack([embeddings_np, context_np])
    y = np.array(labels_list)

    print(f"  Feature matrix: {X_combined.shape}")

    # Train XGBoost (or fallback)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if XGBOOST_AVAILABLE:
        print("  Using XGBoost...")
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            reg_lambda=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
        )
    else:
        print("  Using HistGradientBoosting (fallback)...")
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_depth=6,
            l2_regularization=0.1, random_state=SEED,
        )

    # Cross-validated predictions
    cv_probs = cross_val_predict(clf, X_combined, y, cv=skf, method="predict_proba")[:, 1]

    # Fit final model
    clf.fit(X_combined, y)

    # Evaluation
    auc = roc_auc_score(y, cv_probs)
    acc = accuracy_score(y, (cv_probs >= 0.5).astype(int))
    f1 = f1_score(y, (cv_probs >= 0.5).astype(int))

    print(f"\n  === BASELINE RESULTS (Simple GCN + XGBoost) ===")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Also evaluate GNN-only
    gnn_auc = roc_auc_score(y, gnn_probs)
    print(f"\n  === GNN-ONLY RESULTS ===")
    print(f"  AUC:      {gnn_auc:.4f}")

    return cv_probs, gnn_probs, {"auc": auc, "acc": acc, "f1": f1, "gnn_auc": gnn_auc}


# -------------------------------------------------------
# PHASE D: COMPUTE BASELINE DCI/DIS & SAVE
# -------------------------------------------------------
def compute_baseline_metrics_and_save(cv_probs, meta_list, results):
    """Compute baseline DCI/DIS, save comparison results."""
    print("\n[PHASE D] Computing baseline metrics & saving...")

    rows = []
    for i, m in enumerate(meta_list):
        # Baseline DCI = XGBoost probability (analogous to supervised DCI)
        dci_baseline = cv_probs[i]

        # Baseline DIS = simple heuristic (no latent space)
        # Use the spread of the probability around 0.5 as a proxy
        dis_baseline = 1.0 - abs(cv_probs[i] - 0.5) * 2.0

        rows.append({
            "game_id": m["game_id"],
            "play_id": m["play_id"],
            "dci_baseline": dci_baseline,
            "dis_baseline": dis_baseline,
            "epa": m["epa"],
            "pass_result": m["pass_result"],
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "baseline_gnn_xgboost_metrics.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Save comparison summary
    comparison = {
        "Model": ["SSL R-GCN + HistGBT (Full)", "Simple GCN + XGBoost (Baseline)", "Simple GCN Only"],
        "AUC": ["(run train_dci_head.py)", f"{results['auc']:.4f}", f"{results['gnn_auc']:.4f}"],
        "Note": [
            "512-dim R-GCN + TransformerConv + GRU, SSL pretrained, 400 epochs",
            f"128-dim GCN, supervised, {EPOCHS} epochs + XGBoost",
            f"128-dim GCN only, no boosting",
        ],
    }
    comp_df = pd.DataFrame(comparison)
    comp_path = os.path.join(OUT_DIR, "baseline_comparison_results.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")
    print(f"\n{comp_df.to_string(index=False)}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE: Simple GNN + XGBoost")
    print("=" * 60)
    print(f"Architecture: 2-layer GCN ({IN_DIM} -> {HIDDEN_DIM} -> {EMBED_DIM})")
    print(f"Comparison target: SSL R-GCN + TransformerConv + GRU ({IN_DIM} -> 512 -> 256)")
    print()

    plays_list, labels_list, meta_list = load_and_prepare_data()
    model = train_supervised_gnn(plays_list, labels_list)
    cv_probs, gnn_probs, results = extract_embeddings_and_train_xgb(
        model, plays_list, labels_list, meta_list
    )
    compute_baseline_metrics_and_save(cv_probs, meta_list, results)

    print(f"\n{'='*60}")
    print("BASELINE COMPLETE")
    print(f"{'='*60}")
