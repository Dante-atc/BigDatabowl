#!/usr/bin/env python
# coding: utf-8

"""
Phase 1 — Self-Supervised Backbone Training
===========================================

This module trains the 'DynamicEncoder' model using Self-Supervised Learning (SSL).
It uses Contrastive Learning and Masked Reconstruction to learn physics-aware
player representations without labels.
"""

import os
import math
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# PyG
from torch_geometric.nn import RGCNConv, TransformerConv
from torch_geometric.data import Data

# Import Data Helpers
try:
    from dataset_dynamic import dataloader, build_graphs_from_batch
except ImportError as e:
    raise ImportError("Check dataset_dynamic.py exists.") from e


# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

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
SEED = 42
CHECKPOINT_EVERY = 10

POS_IDX = [0, 1] # x, y

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {device}")


# -----------------------------------------------------------
# AUGMENTATIONS
# -----------------------------------------------------------

def clone_data(g: Data):
    """Safe deep-copy of PyG Data."""
    g2 = Data(x=g.x.clone(), edge_index=g.edge_index.clone())
    if hasattr(g, "edge_type"): g2.edge_type = g.edge_type.clone()
    if hasattr(g, "frame_id"): g2.frame_id = g.frame_id
    if hasattr(g, "play_id"): g2.play_id = g.play_id
    return g2

def node_drop(g: Data, p=NODE_DROP_P):
    n = g.num_nodes
    if p <= 0: return g
    keep_mask = torch.rand(n, device=g.x.device) > p
    if keep_mask.sum() == 0: keep_mask[random.randint(0, n-1)] = True
    
    idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    mapping = -torch.ones(n, dtype=torch.long, device=g.x.device)
    mapping[idx] = torch.arange(idx.size(0), device=g.x.device)
    
    src, dst = g.edge_index
    mask_e = keep_mask[src] & keep_mask[dst]
    new_ei = torch.stack([mapping[src[mask_e]], mapping[dst[mask_e]]], dim=0)
    
    g.x = g.x[idx]
    g.edge_index = new_ei
    if hasattr(g, "edge_type"): g.edge_type = g.edge_type[mask_e]
    return g

def edge_perturb(g: Data, drop_p=EDGE_DROP_P):
    m = g.edge_index.size(1)
    if m == 0 or drop_p <= 0: return g
    keep = torch.rand(m, device=g.edge_index.device) > drop_p
    g.edge_index = g.edge_index[:, keep]
    if hasattr(g, "edge_type"): g.edge_type = g.edge_type[keep]
    return g

def attr_mask_and_jitter(g: Data, mask_p=MASK_PROB, jitter_std=ATTR_JITTER_STD):
    mask = torch.rand_like(g.x) > mask_p
    g.x = g.x * mask.float()
    g.x[:, POS_IDX] += jitter_std * torch.randn_like(g.x[:, POS_IDX])
    return g

def small_rotation_and_translate(g: Data):
    angle = random.uniform(-5.0, 5.0) * math.pi / 180.0
    cosA, sinA = math.cos(angle), math.sin(angle)
    x = g.x[:, POS_IDX].clone()
    center = x.mean(dim=0, keepdim=True)
    x0 = x - center
    R = torch.tensor([[cosA, -sinA],[sinA, cosA]], device=x.device)
    g.x[:, POS_IDX] = (x0 @ R.t()) + center + (0.5 * torch.randn(2, device=x.device))
    return g

def graph_augment(g: Data):
    g2 = clone_data(g)
    g2 = node_drop(g2)
    g2 = edge_perturb(g2)
    g2 = attr_mask_and_jitter(g2)
    g2 = small_rotation_and_translate(g2)
    return g2


# -----------------------------------------------------------
# MODELS
# -----------------------------------------------------------

class DynamicEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations=4, transformer_heads=4):
        super().__init__()
        self.rgcn1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.trans = TransformerConv(hidden_dim, hidden_dim // transformer_heads, heads=transformer_heads)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_type=None, h_prev=None):
        h = F.relu(self.rgcn1(x, edge_index, edge_type))
        h = F.relu(self.rgcn2(h, edge_index, edge_type))
        h = self.trans(h, edge_index)
        if h_prev is not None:
            h = self.gru(h, h_prev)
        return h

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class ReconstructionHead(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


# -----------------------------------------------------------
# LOSSES & TRAINING
# -----------------------------------------------------------

def info_nce_loss(z1, z2, tau=TEMPERATURE):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = (z1 @ z2.T) / max(tau, 1e-6)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return torch.nan_to_num(loss, nan=0.0)

def train_loop(encoder, proj_head, recon_head, dataloader, optimizer, scaler, scheduler, epochs, device):
    encoder.train(); proj_head.train(); recon_head.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", ncols=100)

        for batch in pbar:
            if batch[0] is None: continue
            X_t, X_tp1 = batch
            graphs_t = build_graphs_from_batch(X_t)
            graphs_tp1 = build_graphs_from_batch(X_tp1) # Potentially useful for temporal augs

            optimizer.zero_grad()
            batch_losses = []

            for g in graphs_t:
                g = g.to(device)
                v1, v2 = graph_augment(g), graph_augment(g)
                node_mask = torch.rand(v1.x.size(0), device=device) < MASK_PROB

                with autocast():
                    h1 = encoder(v1.x, v1.edge_index, getattr(v1, "edge_type", None))
                    h2 = encoder(v2.x, v2.edge_index, getattr(v2, "edge_type", None))
                    
                    z1, z2 = proj_head(h1.mean(dim=0, keepdim=True)), proj_head(h2.mean(dim=0, keepdim=True))
                    recon_loss = F.mse_loss(recon_head(h1)[node_mask], v1.x[node_mask]) if node_mask.any() else 0.0
                    batch_losses.append((z1, z2, recon_loss))

            if not batch_losses: continue

            z1_all = torch.cat([b[0] for b in batch_losses], dim=0)
            z2_all = torch.cat([b[1] for b in batch_losses], dim=0)
            r_loss = torch.stack([b[2] for b in batch_losses if isinstance(b[2], torch.Tensor)]).mean()

            with autocast():
                loss = info_nce_loss(z1_all, z2_all) + 0.5 * r_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            total_loss += loss.item()
            n_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / max(1, n_steps)
        loss_history.append(epoch_loss)
        if scheduler: scheduler.step()

        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(encoder.state_dict(), f"{OUT_BACKBONE}.ckpt_epoch{epoch}")
    
    torch.save(encoder.state_dict(), OUT_BACKBONE)
    return loss_history

def main():
    encoder = DynamicEncoder(6, HIDDEN_DIM).to(device)
    proj = ProjectionHead(HIDDEN_DIM, PROJ_DIM).to(device)
    recon = ReconstructionHead(HIDDEN_DIM, 6).to(device)
    
    params = list(encoder.parameters()) + list(proj.parameters()) + list(recon.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    train_loop(encoder, proj, recon, dataloader, optimizer, scaler, scheduler, EPOCHS, device)

if __name__ == "__main__":
    main()