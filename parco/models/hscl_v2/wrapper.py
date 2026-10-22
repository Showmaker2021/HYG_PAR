import torch
import torch.nn as nn
from typing import Optional
from rl4co.utils.ops import gather_by_index
from .layer import CDHypergraphLayer


class CDHGCommunicationWrapper(nn.Module):
    """Hybrid Communication Wrapper: combines PARCO flat communication with CD-HG.

    Implements gated residual:
        h_comm = flat_comm(h) + tanh(g) * (hg_layer(h) - h)
    where g is a learnable scalar parameter initialized to 0. When g=0, the layer
    is bit-exact identical to the original PARCO communication layer.
    """

    def __init__(
        self,
        embed_dim: int,
        config,
        original_layers: Optional[nn.Module] = None,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.original_layers = original_layers

        # Ablation / Configuration Flags
        self.use_gate = getattr(config, "use_gate", True)
        if self.use_gate:
            # Scalar gate parameter initialized strictly to 0
            self.gate = nn.Parameter(torch.zeros(1))

        K = getattr(config, "num_prototypes", getattr(config, "num_clusters", 8))
        tau = getattr(config, "tau", 1.0)
        topk = getattr(config, "topk", 2)
        topk_method = getattr(config, "topk_method", "ste")
        use_prototype_mask = getattr(config, "use_prototype_mask", True)
        parallel_gated_kwargs = kwargs.get("parallel_gated_kwargs", None)
        feedforward_hidden = kwargs.get("feedforward_hidden", None)
        normalization = kwargs.get("normalization", "rms")

        self.cd_layer = CDHypergraphLayer(
            d=embed_dim,
            K=K,
            num_heads=num_heads,
            tau=tau,
            topk=topk,
            topk_method=topk_method,
            use_prototype_mask=use_prototype_mask,
            norm_type=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
            feedforward_hidden=feedforward_hidden,
        )

        # Spatial prior configurations (Requirement D)
        self.use_spatial_prior = getattr(config, "use_spatial_prior", False)
        self.initial_beta = getattr(config, "beta_prior", 1.0)
        self.current_beta = self.initial_beta if self.use_spatial_prior else 0.0

        # Anchor coordinates for spatial prior in [0, 1]^2
        angles = torch.linspace(0, 2 * torch.pi, K + 1)[:K]
        anchors = torch.stack([0.5 + 0.4 * torch.cos(angles), 0.5 + 0.4 * torch.sin(angles)], dim=-1)
        self.register_buffer("spatial_anchors", anchors)

        self.last_H = None
        self.last_p = None

    def reset_history(self):
        self.last_H = None
        self.last_p = None

    def set_beta(self, beta: float):
        """Update beta_t during training warmup/decay."""
        if self.use_spatial_prior:
            self.current_beta = beta
        else:
            self.current_beta = 0.0

    def _compute_spatial_prior(self, td, M: int) -> Optional[torch.Tensor]:
        """Compute distance-based spatial prior pi_ik between agents and spatial anchors."""
        if not self.use_spatial_prior or td is None or not hasattr(td, "get"):
            return None
        locs = td.get("locs", None)
        cur_node = td.get("current_node", None)
        if locs is None or cur_node is None:
            return None
        try:
            cur_coords = gather_by_index(locs, cur_node)  # [B, M, 2]
            # cur_coords: [B, M, 1, 2], anchors: [1, 1, K, 2]
            dists = torch.norm(cur_coords.unsqueeze(2) - self.spatial_anchors.unsqueeze(0).unsqueeze(0), dim=-1) # [B, M, K]
            prior = -dists / 0.5  # normalized negative distance
            return prior
        except Exception:
            return None

    def forward(
        self,
        h: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        td=None,
    ) -> torch.Tensor:
        # 1. Base PARCO flat communication
        if self.original_layers is not None:
            h_flat = self.original_layers(h)
        else:
            h_flat = h

        # 2. Compute tentative action distribution e_bar and intent p
        if embeddings is not None and embeddings.dim() >= 3:
            scale = h.shape[-1] ** 0.5
            logits = torch.matmul(h, embeddings.transpose(-1, -2)) / scale
            if td is not None and hasattr(td, "get"):
                action_mask = td.get("action_mask", None)
                if action_mask is not None:
                    if action_mask.shape[-1] == logits.shape[-1] and action_mask.shape[1] == logits.shape[1]:
                        mask_val = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e9
                        logits = logits.masked_fill(~action_mask.bool(), mask_val)
            p = torch.softmax(logits, dim=-1)
            e_bar = torch.matmul(p, embeddings)
            self.last_p = p
        else:
            e_bar = h
            self.last_p = None

        agent_mask = None
        if td is not None and hasattr(td, "get"):
            action_mask = td.get("action_mask", None)
            if action_mask is not None and action_mask.dim() >= 2:
                agent_mask = action_mask.any(dim=-1).float()

        # 3. Spatial prior computation
        prior = self._compute_spatial_prior(td, h.shape[1]) if self.current_beta > 0.0 else None

        # 4. CD-Hypergraph message passing
        h_hg, H = self.cd_layer(h, e_bar, prior=prior, beta=self.current_beta, mask=agent_mask)
        self.last_H = H

        # 5. Gated residual combination:
        # h_comm = flat_comm(h) + tanh(g) * (hg_layer(h) - h)
        if self.use_gate and hasattr(self, "gate"):
            h_comm = h_flat + torch.tanh(self.gate) * (h_hg - h)
        else:
            h_comm = h_hg

        return h_comm
