#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJ = os.environ.get("PROJ", "/lustre/proyectos/p044/cafa6")

TEST_X_PATH  = f"{PROJ}/embeddings/test/X.npy"
TEST_IDX_PATH  = f"{PROJ}/embeddings/test/index.tsv"
TERMS_PATH = f"{PROJ}/data/raw/train/train_terms.tsv"
MODEL_DIR = f"{PROJ}/models/baseline_linear"
OUT_PATH = f"{PROJ}/results/submission_baseline_tau080.tsv"

EMB_DIM = 480
BATCH_SIZE = 4096

TAU = {"C": 0.80, "F": 0.80, "P": 0.80}
TOPK_PER_ASPECT = 600
MAX_TERMS_PER_PROTEIN = 1500

def canonical_id(pid: str) -> str:
    pid = str(pid)
    if "|" in pid:
        parts = pid.split("|")
        if len(parts) >= 3 and parts[1]:
            return parts[1]
    return pid

class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

def load_vocab_for_aspect(terms_df, aspect):
    return sorted(terms_df[terms_df["aspect"] == aspect]["term"].unique())

def load_model(aspect, out_dim, device):
    ckpt = torch.load(f"{MODEL_DIR}/best_{aspect}.pt", map_location=device)
    model = LinearHead(EMB_DIM, out_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def predict_topk(model, X_memmap, vocab, tau, device):
    N = X_memmap.shape[0]
    out = [[] for _ in range(N)]
    with torch.no_grad():
        for i0 in range(0, N, BATCH_SIZE):
            xb = X_memmap[i0:i0+BATCH_SIZE].astype(np.float32, copy=False)
            xb = torch.from_numpy(xb).to(device, non_blocking=True)
            probs = torch.sigmoid(model(xb)).cpu().numpy()

            for bi in range(probs.shape[0]):
                p = probs[bi]
                idxs = np.where(p >= tau)[0]
                if idxs.size == 0:
                    continue
                if idxs.size > TOPK_PER_ASPECT:
                    sel = np.argpartition(p[idxs], -TOPK_PER_ASPECT)[-TOPK_PER_ASPECT:]
                    idxs = idxs[sel]
                idxs = idxs[np.argsort(p[idxs])[::-1]]
                out[i0 + bi] = [(float(p[j]), vocab[j]) for j in idxs]
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    X_test = np.load(TEST_X_PATH, mmap_mode="r")
    idx_test = pd.read_csv(TEST_IDX_PATH, sep="\t")
    test_ids = [canonical_id(x) for x in idx_test["protein_id"].astype(str).tolist()]

    terms = pd.read_csv(TERMS_PATH, sep="\t")
    terms["aspect"] = terms["aspect"].astype(str)
    terms["term"] = terms["term"].astype(str)

    cand = [[] for _ in range(len(test_ids))]

    for aspect in ["C", "F", "P"]:
        vocab = load_vocab_for_aspect(terms, aspect)
        model = load_model(aspect, len(vocab), device)
        tau = TAU[aspect]
        print(f"[{aspect}] vocab={len(vocab)} tau={tau}")

        topk_lists = predict_topk(model, X_test, vocab, tau, device)
        for i in range(len(cand)):
            if topk_lists[i]:
                cand[i].extend(topk_lists[i])

    # write with cap
    n_lines = 0
    with open(OUT_PATH, "w") as f:
        for pid, c in zip(test_ids, cand):
            if not c:
                continue
            c.sort(key=lambda x: x[0], reverse=True)
            c = c[:MAX_TERMS_PER_PROTEIN]
            for prob, term in c:
                f.write(f"{pid}\t{term}\t{prob:.6f}\n")
                n_lines += 1

    print("[DONE] wrote:", OUT_PATH)
    print("[DONE] lines:", n_lines)

if __name__ == "__main__":
    main()
