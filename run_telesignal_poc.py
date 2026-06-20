"""
run_telesignal_poc.py — Proof-of-concept training run for TeleSignal-Tiny.

Demonstrates the central claim of Section X ("Toward Signal-Native Telecom
Foundation Models"): that self-supervised pretraining DIRECTLY on telecom
KPI time-series (not text) produces representations that transfer better
to a downstream telecom task (anomaly detection) than randomly-initialized
weights of the same architecture, at small scale and on CPU.

Three conditions are compared, all using the IDENTICAL TeleSignalTiny
architecture and the IDENTICAL downstream linear probe:

  A. Random init        — no pretraining, probe trained directly (the
                           "generic, no domain pretraining at all" baseline)
  B. Pretrained (ours)   — self-supervised masked-patch pretraining on
                           UNLABELLED multivariate KPI series, THEN probe
                           trained on top of frozen representations
  C. Pretrained+finetune — same pretrained backbone, but backbone weights
                           are also fine-tuned (not frozen) during the
                           downstream task — the realistic deployment setting

This brackets the question the paper section asks: does signal-native
pretraining buy you anything at all, given that we cannot match Ericsson's
presumed internal scale? If B and C beat A by a clear margin even at this
toy scale, that is real (if modest) evidence for the architectural claim,
clearly distinct from claiming SOTA telecom foundation model performance.

Usage:
    python run_telesignal_poc.py
    python run_telesignal_poc.py --n-cells 60 --n-timesteps 1500 --epochs 15
"""

import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from telekg.pm_simulator import generate_pretraining_corpus
from telekg.telesignal_model import TeleSignalTiny, masked_reconstruction_loss, count_parameters


# ─────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────

def normalise_series(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score normalisation (fit on the whole corpus, standard for this scale)."""
    mean = series.mean(axis=(0, 1), keepdims=True)
    std = series.std(axis=(0, 1), keepdims=True) + 1e-6
    return (series - mean) / std, mean, std


class PretrainDataset(Dataset):
    """Sliding windows over the multivariate KPI corpus, for self-supervised pretraining."""
    def __init__(self, series: np.ndarray, window: int = 64, stride: int = 16):
        self.windows = []
        n_cells, n_t, n_c = series.shape
        for cell_idx in range(n_cells):
            for start in range(0, n_t - window + 1, stride):
                self.windows.append(series[cell_idx, start:start + window, :])
        self.windows = np.stack(self.windows).astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])


class AnomalyProbeDataset(Dataset):
    """
    Windows labelled for the downstream task: does THIS window contain an
    injected anomaly (1) or not (0)? This is the task the linear probe is
    trained on top of the (frozen or fine-tuned) backbone representations.
    """
    def __init__(self, series: np.ndarray, anomaly_mask: np.ndarray, window: int = 64, stride: int = 16):
        self.windows, self.labels = [], []
        n_cells, n_t, n_c = series.shape
        for cell_idx in range(n_cells):
            for start in range(0, n_t - window + 1, stride):
                w = series[cell_idx, start:start + window, :]
                label = int(anomaly_mask[cell_idx, start:start + window].any())
                self.windows.append(w)
                self.labels.append(label)
        self.windows = np.stack(self.windows).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), self.labels[idx]


# ─────────────────────────────────────────────
# Pretraining loop
# ─────────────────────────────────────────────

def pretrain(model: TeleSignalTiny, dataset: PretrainDataset, epochs: int, lr: float = 1e-3,
             batch_size: int = 32, mask_ratio: float = 0.4, verbose=True) -> list[float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            recon, mask, _, patches = model(batch, mask_ratio=mask_ratio)
            loss = masked_reconstruction_loss(recon, patches, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if verbose:
            print(f"  [pretrain] epoch {epoch+1}/{epochs}  masked-patch MSE = {mean_loss:.5f}")
    return losses


# ─────────────────────────────────────────────
# Downstream probe: anomaly detection classifier head
# ─────────────────────────────────────────────

class AnomalyProbe(nn.Module):
    """Linear probe: mean-pool patch embeddings -> single logit (anomaly present in window?)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, embeddings):
        pooled = embeddings.mean(dim=1)  # [batch, d_model]
        return self.head(pooled).squeeze(-1)


def train_probe(backbone: TeleSignalTiny, dataset: AnomalyProbeDataset, epochs: int,
                 freeze_backbone: bool, lr: float = 1e-3, batch_size: int = 32) -> dict:
    """
    Trains a linear probe on top of `backbone` embeddings for anomaly
    detection. If freeze_backbone, only the probe head is trained
    (the standard "linear probing" foundation-model eval protocol).
    Otherwise the backbone is fine-tuned jointly (realistic deployment).
    """
    n = len(dataset)
    split = int(n * 0.7)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    probe = AnomalyProbe(backbone.d_model)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        params = probe.parameters()
    else:
        params = list(backbone.parameters()) + list(probe.parameters())

    optim = torch.optim.AdamW(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        backbone.train(mode=not freeze_backbone)
        probe.train()
        for x, y in train_loader:
            emb = backbone.embed(x)
            logits = probe(emb)
            loss = bce(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    # ── Evaluation ───────────────────────────────────────────────────────
    backbone.eval()
    probe.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            emb = backbone.embed(x)
            logits = probe(emb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds)
            all_labels.append(y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)

    return {
        "n_test": int(len(labels)), "n_positive": int(labels.sum()),
        "precision": round(float(precision), 4), "recall": round(float(recall), 4),
        "f1": round(float(f1), 4), "accuracy": round(float(accuracy), 4),
    }


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=60)
    parser.add_argument("--n-timesteps", type=int, default=1500)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--patch-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{'='*70}\n  TeleSignal-Tiny Proof-of-Concept\n{'='*70}")
    print(f"  Cells: {args.n_cells}  Timesteps/cell: {args.n_timesteps}  "
          f"Total observations: {args.n_cells * args.n_timesteps:,}")

    t0 = time.time()
    print("\n[1/5] Generating synthetic multivariate KPI corpus...")
    corpus = generate_pretraining_corpus(n_cells=args.n_cells, n_timesteps=args.n_timesteps)
    series, anomaly_mask = corpus["series"], corpus["anomaly_mask"]
    n_channels = series.shape[-1]
    print(f"      Shape: {series.shape}  Anomaly rate: {anomaly_mask.mean():.3f}")

    print("\n[2/5] Normalising and building datasets...")
    series_norm, mean, std = normalise_series(series)
    pretrain_ds = PretrainDataset(series_norm, window=args.window, stride=16)
    probe_ds = AnomalyProbeDataset(series_norm, anomaly_mask, window=args.window, stride=16)
    print(f"      Pretrain windows: {len(pretrain_ds)}  Probe windows: {len(probe_ds)} "
          f"(positive rate: {probe_ds.labels.mean():.3f})")

    results = {}

    # ── Condition A: random init, frozen, probe only ──────────────────
    print("\n[3/5] Condition A: Random init (no pretraining) + linear probe...")
    torch.manual_seed(args.seed)
    model_a = TeleSignalTiny(n_channels=n_channels, patch_len=args.patch_len, d_model=args.d_model)
    n_params = count_parameters(model_a)
    res_a = train_probe(model_a, probe_ds, epochs=args.probe_epochs, freeze_backbone=True)
    print(f"      {res_a}")
    results["A_random_init_frozen"] = res_a

    # ── Condition B: pretrained, frozen, probe only ────────────────────
    print(f"\n[4/5] Condition B: Self-supervised pretraining ({args.pretrain_epochs} epochs) "
          f"then frozen probe...")
    torch.manual_seed(args.seed)
    model_b = TeleSignalTiny(n_channels=n_channels, patch_len=args.patch_len, d_model=args.d_model)
    pretrain_losses = pretrain(model_b, pretrain_ds, epochs=args.pretrain_epochs)
    res_b = train_probe(model_b, probe_ds, epochs=args.probe_epochs, freeze_backbone=True)
    print(f"      {res_b}")
    results["B_pretrained_frozen"] = res_b
    results["pretrain_loss_curve"] = [round(l, 5) for l in pretrain_losses]

    # ── Condition C: pretrained, fine-tuned end-to-end ──────────────────
    print(f"\n[5/5] Condition C: Pretrained backbone + end-to-end fine-tuning on probe task...")
    torch.manual_seed(args.seed)
    model_c = TeleSignalTiny(n_channels=n_channels, patch_len=args.patch_len, d_model=args.d_model)
    pretrain(model_c, pretrain_ds, epochs=args.pretrain_epochs, verbose=False)
    res_c = train_probe(model_c, probe_ds, epochs=args.probe_epochs, freeze_backbone=False)
    print(f"      {res_c}")
    results["C_pretrained_finetuned"] = res_c

    elapsed = time.time() - t0

    print(f"\n{'='*70}\n  Summary\n{'='*70}")
    print(f"  Model parameters:                {n_params:,}")
    print(f"  Total wall-clock time:           {elapsed:.1f}s  (CPU)")
    print(f"  {'Condition':<32}{'F1':>8}{'Precision':>12}{'Recall':>10}{'Accuracy':>10}")
    for name, key in [("A: Random init (frozen)", "A_random_init_frozen"),
                       ("B: Pretrained (frozen)", "B_pretrained_frozen"),
                       ("C: Pretrained (fine-tuned)", "C_pretrained_finetuned")]:
        r = results[key]
        print(f"  {name:<32}{r['f1']:>8}{r['precision']:>12}{r['recall']:>10}{r['accuracy']:>10}")

    f1_a, f1_b = results["A_random_init_frozen"]["f1"], results["B_pretrained_frozen"]["f1"]
    rel_improvement = ((f1_b - f1_a) / f1_a * 100) if f1_a > 0 else float("inf")
    print(f"\n  Pretraining vs random-init F1 improvement: {rel_improvement:+.1f}%")
    print(f"\n  NOTE: small-scale (~{n_params/1e6:.2f}M params, {args.n_cells*args.n_timesteps:,} "
          f"observations, CPU-only). This is a proof of concept for the architectural\n"
          f"  claim in the paper, not a competitor to an industrial-scale telecom foundation model.")

    results["meta"] = {
        "n_params": n_params, "elapsed_seconds": round(elapsed, 1),
        "n_cells": args.n_cells, "n_timesteps": args.n_timesteps,
        "total_observations": args.n_cells * args.n_timesteps,
        "f1_relative_improvement_pct": round(rel_improvement, 2) if f1_a > 0 else None,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results written to {args.json}")


if __name__ == "__main__":
    main()
