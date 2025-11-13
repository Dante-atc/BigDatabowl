#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import math
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# PyG
from torch_geometric.nn import RGCNConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data


# # General Configuration

# In[ ]:


OUT_BACKBONE = "/lustre/home/dante/compartido/models/backbone_ssl.pth"
OUT_LOSS_PLOT = "/lustre/home/dante/compartido/models/train_loss.png"

EPOCHS = 400
BATCH_SIZE = 1024
LR = 5e-5
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 512
PROJ_DIM = 256
MASK_PROB = 0.25
EDGE_DROP_P = 0.1
NODE_DROP_P = 0.15
ATTR_JITTER_STD = 0.02
TEMPERATURE = 0.1
GRAD_CLIP = 2.0
NUM_WORKERS = 8
SEED = 42
CHECKPOINT_EVERY = 10  # epochs

# Feature indices (assume node features are [x,y,s,a,o,dir] in that order)
POS_IDX = [0, 1]

# Seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] running on device: {device}")

# Import dataset helpers
try:
    # dataset_dynamic must expose: dataloader, build_graphs_from_batch
    from dataset_dynamic import dataloader, build_graphs_from_batch
except Exception as e:
    raise ImportError("Could not import dataloader/build_graphs_from_batch from dataset_dynamic.py: " + str(e))


# # Augmentations

# In[ ]:


def clone_data(g: Data):
    """Deep-copy a PyG Data object (safe to mutate)."""
    g2 = Data(
        x=g.x.clone() if g.x is not None else None,
        edge_index=g.edge_index.clone() if g.edge_index is not None else None,
    )
    for k, v in g:
        if k in ("x", "edge_index"):
            continue
        g2[k] = v
    # copy edge_type if exists
    if hasattr(g, "edge_type"):
        g2.edge_type = g.edge_type.clone()
    if hasattr(g, "frame_id"):
        g2.frame_id = g.frame_id
    if hasattr(g, "play_id"):
        g2.play_id = g.play_id
    return g2

def node_drop(g: Data, p=NODE_DROP_P):
    n = g.num_nodes
    if p <= 0:
        return g
    keep_mask = torch.rand(n, device=g.x.device) > p
    if keep_mask.all().item() is False and keep_mask.sum().item() == 0:
        # keep at least 1
        keep_mask[random.randint(0, n-1)] = True
    idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    mapping = -torch.ones(n, dtype=torch.long, device=g.x.device)
    mapping[idx] = torch.arange(idx.size(0), device=g.x.device)
    # filter edges
    ei = g.edge_index
    src, dst = ei
    mask_e = keep_mask[src] & keep_mask[dst]
    new_ei = torch.stack([mapping[src[mask_e]], mapping[dst[mask_e]]], dim=0)
    g.x = g.x[idx]
    g.edge_index = new_ei
    if hasattr(g, "edge_type"):
        g.edge_type = g.edge_type[mask_e]
    return g

def edge_perturb(g: Data, drop_p=EDGE_DROP_P):
    m = g.edge_index.size(1)
    if m == 0 or drop_p <= 0:
        return g
    keep = torch.rand(m, device=g.edge_index.device) > drop_p
    g.edge_index = g.edge_index[:, keep]
    if hasattr(g, "edge_type"):
        g.edge_type = g.edge_type[keep]
    return g

def attr_mask_and_jitter(g: Data, mask_p=MASK_PROB, jitter_std=ATTR_JITTER_STD):
    # mask some node features and jitter positions
    mask = torch.rand_like(g.x) > mask_p
    g.x = g.x * mask.float()
    # jitter positions (first two cols)
    g.x[:, POS_IDX] = g.x[:, POS_IDX] + jitter_std * torch.randn_like(g.x[:, POS_IDX])
    return g

def small_rotation_and_translate(g: Data, max_angle_deg=5.0, translate_scale=0.5):
    # rotate around center (approx), small angle in degrees
    angle = (random.uniform(-max_angle_deg, max_angle_deg) * math.pi / 180.0)
    cosA, sinA = math.cos(angle), math.sin(angle)
    x = g.x[:, POS_IDX].clone()
    # center
    center = x.mean(dim=0, keepdim=True)
    x0 = x - center
    R = torch.tensor([[cosA, -sinA],[sinA, cosA]], device=x.device)
    xr = (x0 @ R.t()) + center
    g.x[:, POS_IDX] = xr
    # small translate
    trans = (translate_scale * torch.randn(2, device=x.device))
    g.x[:, POS_IDX] += trans
    return g

def graph_augment(g: Data):
    g2 = clone_data(g)
    # Apply sequence of augmentations (order matters a bit)
    g2 = node_drop(g2, NODE_DROP_P)
    g2 = edge_perturb(g2, EDGE_DROP_P)
    g2 = attr_mask_and_jitter(g2, MASK_PROB, ATTR_JITTER_STD)
    g2 = small_rotation_and_translate(g2)
    return g2


# # Model: Encoder + proj + recon

# In[ ]:


class DynamicEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations=4, transformer_heads=4):
        super().__init__()
        # two-layer RGCN (relational)
        self.rgcn1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        # transformer conv for global attention
        self.trans = TransformerConv(hidden_dim, hidden_dim // transformer_heads, heads=transformer_heads)
        # GRUCell for per-node temporal update (we use small state per node)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type=None, h_prev=None):
        # x: [N, F], edge_index: [2, E]
        h = F.relu(self.rgcn1(x, edge_index, edge_type))
        h = F.relu(self.rgcn2(h, edge_index, edge_type))
        # transformer expects x and edge_index; returns [N, out]
        h = self.trans(h, edge_index)
        # temporal update: if h_prev provided (node states), apply GRUCell per node
        if h_prev is not None:
            # h_prev and h must be of same nodes; assume same ordering
            h = self.gru(h, h_prev)
        return h  # [N, hidden_dim]

DynamicGraphEncoder = DynamicEncoder

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
    def forward(self, x):
        # x: [num_nodes, hidden_dim] or [B, hidden_dim] (if pooled)
        z = self.net(x)
        return F.normalize(z, dim=-1)

class ReconstructionHead(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# # Loss Functions

# In[ ]:


def info_nce_loss(z1, z2, tau=TEMPERATURE):
    # z1, z2: [B, D]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.T) / max(tau, 1e-6)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss1 + loss2)

    if not torch.isfinite(loss):
        print("[WARN] NaN detected in InfoNCE, replacing with 0")
        loss = torch.tensor(0.0, device=z1.device)

    return loss


def masked_recon_loss(pred, target, node_mask):
    # pred: [N, F], target: [N, F], node_mask: boolean [N] selects nodes to compute loss
    if node_mask is None:
        return F.mse_loss(pred, target)
    if node_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[node_mask], target[node_mask])


# # Model Initialization

# In[ ]:


# Ajusta a tus features
IN_DIM = 6      # x, y, s, a, o, dir
HIDDEN_DIM = 128
OUTPUT_DIM = 6
PROJ_DIM = 128

encoder = DynamicGraphEncoder(IN_DIM, HIDDEN_DIM).to(device)
projection = ProjectionHead(HIDDEN_DIM, PROJ_DIM).to(device)
reconstruction = ReconstructionHead(HIDDEN_DIM, OUTPUT_DIM).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + 
    list(projection.parameters()) + 
    list(reconstruction.parameters()),
    lr=1e-4
)


# # Training

# In[ ]:


def train_loop(encoder, proj_head, recon_head, dataloader,
               optimizer, scaler, scheduler=None, epochs=EPOCHS, device=device):

    encoder.train()
    proj_head.train()
    recon_head.train()

    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", ncols=120)
        for batch in pbar:
            # batch expected from dataset_dynamic.dataloader:
            # either lists of Data objects per batch, or tensors to convert via build_graphs_from_batch
            if batch[0] is None:
                continue
            X_t_batch, X_tp1_batch = batch  # these are tensors or already preprocessed
            graphs_t = build_graphs_from_batch(X_t_batch)
            graphs_tp1 = build_graphs_from_batch(X_tp1_batch)

            # process per sample in batch (we can vectorize later if needed)
            batch_losses = []
            optimizer.zero_grad()
            for g_t, g_tp1 in zip(graphs_t, graphs_tp1):
                # ensure on CPU->GPU transfer
                g_t = g_t.to(device)
                g_tp1 = g_tp1.to(device)

                # create two augmented views from g_t
                v1 = graph_augment(g_t)
                v2 = graph_augment(g_t)

                # optional: small chance to also use g_tp1 as view to inject temporal signal
                if random.random() < 0.1:
                    v2 = graph_augment(g_tp1)

                # node feature masks for reconstruction: choose nodes to mask (boolean per node)
                node_mask = (torch.rand(v1.x.size(0), device=v1.x.device) < MASK_PROB)

                with autocast():
                    # get node embeddings
                    # no previous hidden state tracked across batches — here we use g_tp1 embedding as pseudo-h_prev
                    h_v1 = encoder(v1.x, v1.edge_index, getattr(v1, "edge_type", None), None)
                    h_v2 = encoder(v2.x, v2.edge_index, getattr(v2, "edge_type", None), None)

                    # pool graph-level embeddings (mean pool)
                    g_emb1 = h_v1.mean(dim=0, keepdim=True)  # [1, hidden]
                    g_emb2 = h_v2.mean(dim=0, keepdim=True)

                    # projection for contrastive
                    z1 = proj_head(g_emb1)  # [1, proj_dim]
                    z2 = proj_head(g_emb2)

                    # reconstruction predictions for masked nodes (use v1)
                    node_preds = recon_head(h_v1)  # [N, feat_dim]
                    recon_loss = masked_recon_loss(node_preds, v1.x, node_mask)

                    # contrastive loss expects batch >1; we will accumulate small batches and compute symmetric NCE across list
                    # For simplicity we will collect g_embs per micro-batch and compute NCE outside loop.
                    batch_losses.append((z1, z2, recon_loss))

            # Now compute contrastive over micro-batch of embeddings
            if len(batch_losses) == 0:
                continue

            # Unzip
            z1_list = torch.cat([t[0] for t in batch_losses], dim=0)  # [M, D]
            z2_list = torch.cat([t[1] for t in batch_losses], dim=0)
            recon_losses = torch.stack([t[2] for t in batch_losses], dim=0).mean()

            with autocast():
                loss_contrast = info_nce_loss(z1_list, z2_list)
                loss_total = loss_contrast + 0.5 * recon_losses

            if not torch.isfinite(loss_total):
                print(f"[WARN] NaN detected at epoch {epoch}, skipping batch")
                continue

            scaler.scale(loss_total).backward()
            # gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss_total.item()
            n_steps += 1
            pbar.set_postfix({"loss": f"{(total_loss / n_steps):.4f}"})

        epoch_loss = total_loss / max(1, n_steps)
        loss_history.append(epoch_loss)
        print(f"[Epoch {epoch}] avg loss: {epoch_loss:.4f}")

        # checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            # save a partial checkpoint of encoder weights
            os.makedirs(os.path.dirname(OUT_BACKBONE), exist_ok=True)
            torch.save(encoder.state_dict(), OUT_BACKBONE + f".ckpt_epoch{epoch}")
            print(f"[INFO] checkpoint saved: {OUT_BACKBONE}.ckpt_epoch{epoch}")

        # scheduler step (actualiza el learning rate)
        if 'scheduler' in locals():
            scheduler.step()

    # final save
    os.makedirs(os.path.dirname(OUT_BACKBONE), exist_ok=True)
    torch.save(encoder.state_dict(), OUT_BACKBONE)
    print(f"[INFO] Final backbone saved to {OUT_BACKBONE}")

    # plot loss
    try:
        plt.figure(figsize=(8,4))
        plt.plot(loss_history, label="train_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_LOSS_PLOT)
        print(f"[INFO] Loss plot saved to {OUT_LOSS_PLOT}")
    except Exception as e:
        print("[WARN] Could not save loss plot:", e)

    return loss_history


# # Main

# In[ ]:


def main():
    # build models
    encoder = DynamicEncoder(in_dim=6, hidden_dim=HIDDEN_DIM).to(device)
    proj_head = ProjectionHead(HIDDEN_DIM, PROJ_DIM).to(device)
    recon_head = ReconstructionHead(HIDDEN_DIM, out_dim=6).to(device)

    params = list(encoder.parameters()) + list(proj_head.parameters()) + list(recon_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ✅ nuevo: scheduler suave para estabilizar el entrenamiento
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = GradScaler()

    print("[INFO] Starting training...")
    losses = train_loop(
        encoder, proj_head, recon_head,
        dataloader, optimizer, scaler,
        scheduler=scheduler,   # ✅ pasa el scheduler
        epochs=EPOCHS, device=device
    )
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()

