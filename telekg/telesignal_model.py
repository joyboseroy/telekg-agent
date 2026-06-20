"""
TeleSignal-Tiny: A Small Signal-Native Transformer for Telecom KPI Time-Series

This is a proof-of-concept "telecom foundation model," in the sense Ericsson's
described initiative uses the term: a transformer pretrained DIRECTLY on
streaming KPI/PM-counter time-series, not on text. The pretraining objective
is masked patch reconstruction (the time-series analogue of BERT's masked
language modelling / PatchTST's self-supervised objective):

  1. A multivariate KPI time-series is divided into non-overlapping patches
     along the time axis (e.g. 8 consecutive timesteps per patch).
  2. Each patch, across all KPI channels, is linearly projected into a
     "token" embedding — this is the signal-domain analogue of a word
     embedding: instead of looking up a vocabulary entry, we project a
     numeric vector.
  3. A random subset of patches is masked (replaced with a learned [MASK]
     token), and the transformer is trained to reconstruct the masked
     patches from the unmasked context — exactly BERT's MLM objective,
     applied to numbers instead of words.
  4. After pretraining, the learned representations can be probed for
     downstream tasks (anomaly detection, forecasting) via a small linear
     head, without retraining the backbone — the standard
     foundation-model evaluation protocol.

SCALE AND HONESTY NOTE: This is intentionally small (under 2M parameters,
CPU-trainable in minutes on a few hundred thousand timesteps of synthetic
data). It is a proof of concept that signal-native pretraining is feasible
and produces transferable representations, NOT a claim of competing with
an industrial-scale telecom foundation model trained on real multi-vendor
network data with GPU clusters. The paper section this supports is explicit
about this scale gap.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Patch embedding (the signal-domain analogue of token embedding)
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Projects a patch of [patch_len, n_channels] raw KPI values into a
    single d_model-dimensional embedding. This plays the same role as a
    word embedding lookup table, but for continuous multivariate signal
    patches instead of discrete word IDs.
    """
    def __init__(self, patch_len: int, n_channels: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.proj = nn.Linear(patch_len * n_channels, d_model)

    def forward(self, x):
        # x: [batch, n_patches, patch_len, n_channels]
        b, n, p, c = x.shape
        x = x.reshape(b, n, p * c)
        return self.proj(x)  # [batch, n_patches, d_model]


class PatchUnembedding(nn.Module):
    """Inverse of PatchEmbedding — reconstructs raw values from d_model embedding."""
    def __init__(self, patch_len: int, n_channels: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.proj = nn.Linear(d_model, patch_len * n_channels)

    def forward(self, x):
        # x: [batch, n_patches, d_model]
        b, n, d = x.shape
        out = self.proj(x)
        return out.reshape(b, n, self.patch_len, self.n_channels)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the patch sequence."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────

class TeleSignalTiny(nn.Module):
    """
    Small transformer encoder for multivariate telecom KPI time-series.

    Pretraining objective: masked patch reconstruction (MSE loss on masked
    patches only, following the BERT-style "predict only the masked tokens"
    convention).
    """
    def __init__(self, n_channels: int, patch_len: int = 8, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dim_ff: int = 128,
                 max_patches: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(patch_len, n_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_patches)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = PatchUnembedding(patch_len, n_channels, d_model)

        self.n_params = sum(p.numel() for p in self.parameters())

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_timesteps, n_channels] -> [batch, n_patches, patch_len, n_channels]"""
        b, t, c = x.shape
        n_patches = t // self.patch_len
        x = x[:, :n_patches * self.patch_len, :]
        return x.reshape(b, n_patches, self.patch_len, c)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.4):
        """
        x: [batch, n_timesteps, n_channels] raw (normalised) KPI values.
        Returns: reconstruction [batch, n_patches, patch_len, n_channels],
                 mask [batch, n_patches] (bool, True = was masked),
                 embeddings [batch, n_patches, d_model] (for downstream probing)
        """
        patches = self.patchify(x)                    # [b, n_patches, patch_len, c]
        b, n_patches = patches.shape[0], patches.shape[1]

        tokens = self.patch_embed(patches)              # [b, n_patches, d_model]
        tokens = self.pos_enc(tokens)

        mask = torch.rand(b, n_patches, device=x.device) < mask_ratio
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token

        encoded = self.encoder(masked_tokens)            # [b, n_patches, d_model]
        recon = self.unembed(encoded)                    # [b, n_patches, patch_len, c]

        return recon, mask, encoded, patches

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get representations with no masking, for downstream probing."""
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)
        tokens = self.pos_enc(tokens)
        return self.encoder(tokens)  # [b, n_patches, d_model]


def masked_reconstruction_loss(recon, patches, mask):
    """MSE loss computed ONLY over masked patches (BERT-style)."""
    # recon, patches: [b, n_patches, patch_len, c] ; mask: [b, n_patches]
    diff2 = (recon - patches) ** 2
    diff2 = diff2.mean(dim=(-1, -2))      # [b, n_patches] — per-patch MSE
    masked_loss = (diff2 * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return masked_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
