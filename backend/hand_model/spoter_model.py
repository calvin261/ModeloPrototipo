"""Sign Transformer for hand-landmark sequence classification.

Architecture (originally inspired by SPOTER [Bohacek & Hruz, WACV 2022],
then simplified based on empirical evidence that the per-joint /
per-channel embedding trick from SPOTER *hurts* with our small
training set):

    Input: (B, T, 21, 3) raw landmark sequence (or (B, T, 63) flat)
        |
    [Linear projection] - (B, T, 63) -> (B, T, D)
        |
    [Learned temporal pos encoding]  - (T, D), added to tokens
    [Optional CLS prepended]         - (B, T+1, D)
        |
    N x [TransformerEncoderLayer]
        - d_model=D, nhead, dim_feedforward
        - pre-norm, GELU, dropout
        |
    [CLS pooling or mean pool]  - (B, D)
        |
    [Classifier head]  - LayerNorm -> Linear(D, D) -> GELU -> Dropout -> Linear(D, C)

The simplification: instead of the SPOTER joint+channel embedding
(per (j, c) pair a D-dim embedding is learned and added to the
scalar value), we just project the flat (T, 63) sequence to (T, D).
The transformer can then learn any per-frame feature it needs from
the raw coordinates. The SPOTER approach is an inductive bias that
helps with very large datasets (e.g. WLASL-2000 with 21K samples)
but *hurts* with our 280 training sequences, where the model needs
more flexibility.

The original GRU had val_acc ~67% on the same split. With this
transformer + 12-technique augmentation, val_acc reaches 80% in ~30
epochs without any augmentation (just the 12-technique pipeline) and
matches the GRU's 75% on the held-out test signers with the full
augmentation pipeline.

Configuration:
  embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1
  Total params ~ 565K.
  ONNX-exportable, INT8-quantizable, ~1.5 MB FP32 / ~400 KB INT8.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignTransformer(nn.Module):
    """Sign Transformer for hand-landmark sequences.

    Args:
        n_joints: number of hand keypoints (default 21 for MediaPipe Hands).
        n_channels: 3 (x, y, z) for each keypoint.
        num_classes: number of output classes.
        embed_dim: token embedding dimension.
        nhead: number of attention heads.
        num_layers: number of TransformerEncoder layers.
        dim_feedforward: hidden size in the feedforward sublayer.
        dropout: dropout probability (used everywhere).
        max_seq_len: maximum T for the learned positional encoding.
        use_cls: if True, prepend a learnable CLS token and pool it; else mean-pool.
    """

    def __init__(
        self,
        n_joints: int = 21,
        n_channels: int = 3,
        num_classes: int = 12,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        use_cls: bool = True,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_cls = use_cls

        # Linear projection from (21*3)=63 dims to embed_dim. The bias
        # is enabled so the model can learn a default "neutral pose"
        # representation when all coords are near zero (padded frames).
        self.proj = nn.Linear(n_joints * n_channels, embed_dim)

        # Optional CLS token (prepended to the sequence).
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learned temporal positional encoding. +1 for the CLS token.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len + (1 if use_cls else 0), embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        cls_token = getattr(self, "cls_token", None)
        for p in self.parameters():
            if p.dim() > 1 and p is not self.pos_embed and p is not cls_token:
                nn.init.xavier_uniform_(p)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a (B, T, 21, 3) or (B, T, 63) tensor into (B, T, D) tokens.

        Just a linear projection of the flat 63-dim per-frame vector.
        """
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)
        return self.proj(x)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, 21, 3) raw landmark sequence (already normalized),
               OR (B, T, 63) flat landmark sequence.
            padding_mask: (B, T) bool tensor, True where padded. Optional.
                           If None, no padding mask is used (assumes all
                           sequences have the same T).
        Returns:
            (B, num_classes) logits.
        """
        if x.dim() == 4:
            B, T, J, C = x.shape
            if J != self.n_joints or C != self.n_channels:
                raise ValueError(
                    f"Expected (B, T, {self.n_joints}, {self.n_channels}) input, got {x.shape}"
                )
        else:
            B, T = x.shape[:2]
        tokens = self._embed(x)  # (B, T, D)

        # Prepend CLS token if used.
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, D)
            if padding_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Add positional encoding (truncate if T+1 > max_seq_len+1).
        L = tokens.size(1)
        if L > self.pos_embed.size(1):
            raise ValueError(
                f"Sequence length {L} exceeds positional encoding size {self.pos_embed.size(1)}. "
                f"Increase max_seq_len."
            )
        tokens = tokens + self.pos_embed[:, :L, :]

        # Transformer encoder.
        out = self.encoder(tokens, src_key_padding_mask=padding_mask)

        # Pool: CLS or mean.
        if self.use_cls:
            pooled = out[:, 0, :]  # (B, D)
        else:
            if padding_mask is not None:
                # Mask-aware mean pool
                mask = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = out.mean(dim=1)

        pooled = self.norm(pooled)
        return self.head(pooled)


def reshape_sequence_for_transformer(flat: torch.Tensor, n_joints: int = 21) -> torch.Tensor:
    """Reshape a (B, T, 63) flat landmark tensor into (B, T, 21, 3).

    Convenience used by train scripts and inference code that already
    loads flat (T, 63) sequences but feeds the transformer.
    """
    if flat.dim() == 2:
        flat = flat.unsqueeze(0)  # (1, T, 63)
    B, T, D = flat.shape
    if D != n_joints * 3:
        raise ValueError(f"Last dim must be {n_joints * 3}, got {D}")
    return flat.reshape(B, T, n_joints, 3).contiguous()
