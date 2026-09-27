from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.mlp import MLP

from parco.models.nn.transformer import Normalization


class HypergraphAttentionConv(nn.Module):
    """Hypergraph Attention Convolution module for HSCL.

    Implements a two-phase hypergraph communication mechanism:
      1. Intra-group Attention: Attention between agents belonging to the same cluster.
      2. Inter-group Attention: Attention between cluster centroids across top-r closest clusters,
         then broadcasted back to individual agents.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        inter_group_top_r: int = 2,
        feedforward_hidden: Optional[int] = None,
        normalization: Optional[str] = "instance",
        norm_after: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.inter_group_top_r = inter_group_top_r
        self.norm_after = norm_after

        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []

        self.intra_attn = MultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.inter_attn = MultiHeadAttention(embed_dim, num_heads, bias=bias)

        self.norm_intra = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else nn.Identity()
        )
        self.norm_inter = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else nn.Identity()
        )
        self.ffn = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_neurons=num_neurons,
            hidden_act="ReLU",
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, cluster_assignment: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for HypergraphAttentionConv.

        Args:
            x: Agent embeddings of shape [B, M, d]
            cluster_assignment: Cluster indices for each agent of shape [B, M]

        Returns:
            Updated agent representations of shape [B, M, d]
        """
        B, M, d = x.shape
        if M <= 1:
            # Single agent: bypass hypergraph partitioning
            return x

        num_clusters = int(cluster_assignment.max().item()) + 1
        num_clusters = min(num_clusters, M)

        # ---------------- Phase 1: Intra-group Attention ----------------
        # Mask: True where agents belong to the same cluster
        intra_mask = (
            cluster_assignment.unsqueeze(-1) == cluster_assignment.unsqueeze(-2)
        )  # [B, M, M]

        normed_x = self.norm_intra(x) if not self.norm_after else x
        intra_out = self.intra_attn(normed_x, attn_mask=intra_mask)

        # ---------------- Phase 2: Inter-group Attention ----------------
        if num_clusters > 1:
            # Construct one-hot membership [B, M, C]
            membership = F.one_hot(
                cluster_assignment, num_classes=num_clusters
            ).to(x.dtype)
            counts = membership.sum(dim=1).clamp_min(1).unsqueeze(-1)  # [B, C, 1]

            # Cluster centroids via mean-pooling [B, C, d]
            centroids = torch.einsum("bmc,bmd->bcd", membership, x) / counts

            # Inter-cluster distance matrix [B, C, C]
            dist = (
                (centroids.unsqueeze(2) - centroids.unsqueeze(1))
                .square()
                .sum(dim=-1)
            )

            r = min(self.inter_group_top_r, num_clusters)
            top_r_indices = dist.topk(r, dim=-1, largest=False).indices  # [B, C, r]

            inter_mask = torch.zeros(
                B, num_clusters, num_clusters, dtype=torch.bool, device=x.device
            )
            inter_mask.scatter_(-1, top_r_indices, True)
            inter_mask.diagonal(dim1=-2, dim2=-1).fill_(True)

            normed_centroids = (
                self.norm_inter(centroids) if not self.norm_after else centroids
            )
            attended_centroids = self.inter_attn(
                normed_centroids, attn_mask=inter_mask
            )

            # Broadcast back to individual agents [B, M, d]
            inter_out = torch.einsum("bmc,bcd->bmd", membership, attended_centroids)
        else:
            inter_out = 0

        # ---------------- Combine with Residuals & FFN ----------------
        if not self.norm_after:
            h = x + intra_out + inter_out
            h = h + self.ffn(self.norm_ffn(h))
        else:
            h = self.norm_intra(x + intra_out + inter_out)
            h = self.norm_ffn(h + self.ffn(h))

        return h
