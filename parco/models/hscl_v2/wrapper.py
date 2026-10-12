import torch
import torch.nn as nn
from typing import Optional
from .layer import CDHypergraphLayer


class CDHGCommunicationWrapper(nn.Module):
    """Wrapper around CDHypergraphLayer that adapts to context embedding calls.

    Computes tentative action proposals e_bar on-the-fly from context_embed
    and encoder node embeddings, then runs conflict-driven hypergraph message passing.
    """

    def __init__(self, embed_dim: int, config, num_heads: int = 4, **kwargs):
        super().__init__()
        self.config = config
        K = getattr(config, "num_prototypes", getattr(config, "num_clusters", 8))
        tau = getattr(config, "tau", 1.0)
        topk = getattr(config, "topk", 2)
        self.cd_layer = CDHypergraphLayer(
            d=embed_dim,
            K=K,
            num_heads=num_heads,
            tau=tau,
            topk=topk,
        )
        self.last_H = None
        self.last_p = None

    def reset_history(self):
        self.last_H = None
        self.last_p = None

    def forward(
        self,
        h: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        td=None,
    ) -> torch.Tensor:
        if embeddings is not None and embeddings.dim() >= 3:
            scale = h.shape[-1] ** 0.5
            # h: [B, M, d], embeddings: [B, N, d]
            logits = torch.matmul(h, embeddings.transpose(-1, -2)) / scale
            if td is not None:
                action_mask = td.get("action_mask", None) if hasattr(td, "get") else None
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

        h_out, H = self.cd_layer(h, e_bar, mask=agent_mask)
        self.last_H = H
        return h_out
