from typing import Optional, Union

import torch
import torch.nn as nn

from .config import HSCLConfig
from .eigensolver import AutoEigenSolver
from .history import StateHistoryBuffer
from .hypergraph_conv import HypergraphAttentionConv
from .spectral import affinity_from_history, batched_kmeans


class HSCLCommunicationLayer(nn.Module):
    """Hypergraph-Structured Communication Layer (HSCL).

    Wraps the communication mechanism with:
      - Fallback: Bit-exact execution of the original PARCO CommunicationLayer when use_hscl=False.
      - HSCL Mode: Dynamic spectral graph partitioning + HypergraphAttentionConv.
    """

    def __init__(
        self,
        original_layers: Union[nn.Module, nn.Sequential],
        embed_dim: int = 128,
        num_heads: int = 8,
        config: Optional[HSCLConfig] = None,
        normalization: Optional[str] = "instance",
        norm_after: bool = False,
        feedforward_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or HSCLConfig()
        self.original_layers = original_layers
        self.embed_dim = embed_dim

        self.history_buffer = StateHistoryBuffer(
            window_size=self.config.history_window
        )
        self.eigensolver = AutoEigenSolver(self.config)
        self.hypergraph_conv = HypergraphAttentionConv(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inter_group_top_r=self.config.inter_group_top_r,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            norm_after=norm_after,
        )

    def reset_history(self):
        """Reset internal history buffer for new rollout/episode."""
        self.history_buffer.reset()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Agent embeddings of shape [B, M, d]
            mask: Optional attention mask for original layer compatibility

        Returns:
            Processed agent representations of shape [B, M, d]
        """
        # 1. Fallback mode: bit-exact execution of original layer
        if not self.config.use_hscl:
            if isinstance(self.original_layers, nn.Sequential):
                return self.original_layers(x)
            else:
                return (
                    self.original_layers(x, mask=mask)
                    if mask is not None
                    else self.original_layers(x)
                )

        # 2. HSCL mode
        history = self.history_buffer.update(x)

        with torch.no_grad():
            L = affinity_from_history(
                history, knn=self.config.knn, sigma=self.config.sigma
            )
            eigvecs = self.eigensolver.smallest_k_eigenvectors(
                L, self.config.num_clusters
            )
            cluster_assignment = batched_kmeans(eigvecs, self.config.num_clusters)

        # Apply Hypergraph Attention Convolution
        out = self.hypergraph_conv(x, cluster_assignment)
        return out
