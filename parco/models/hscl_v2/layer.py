import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


from parco.models.nn.transformer import ParallelGatedMLP


class CDHypergraphLayer(nn.Module):
    """Conflict-Driven Hypergraph Layer (CD-HG v2.2).

    Replaces flat pairwise all-to-all attention with a differentiable,
    soft-overlapping hypergraph message-passing mechanism followed by
    a Transformer-style Feed-Forward Network (FFN).

    Args:
        d (int): Hidden feature dimension.
        K (int): Number of hyperedge prototypes (K_max).
        num_heads (int): Number of attention heads for inter-hyperedge MHA.
        tau (float): Temperature for prototype similarity.
        topk (int): Sparsity constraint (number of hyperedges each agent can belong to).
        dropout (float): Dropout probability.
        norm_type (str): Normalization type: 'rms' or 'layer'.
        parallel_gated_kwargs (Optional[dict]): ParallelGatedMLP kwargs (e.g. SiLU activation).
        feedforward_hidden (Optional[int]): Hidden dimension of FFN if not gated.
    """

    def __init__(
        self,
        d: int,
        K: int = 8,
        num_heads: int = 4,
        tau: float = 1.0,
        topk: int = 2,
        topk_method: str = "ste",
        use_prototype_mask: bool = True,
        eps_proto: float = 1e-4,
        aux_detach_input: bool = True,
        dropout: float = 0.0,
        norm_type: str = "rms",
        parallel_gated_kwargs: Optional[dict] = None,
        feedforward_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.tau = tau
        self.topk = topk
        self.topk_method = topk_method
        self.use_prototype_mask = use_prototype_mask
        self.eps_proto = eps_proto
        self.aux_detach_input = aux_detach_input

        # 1. K Learnable Hyperedge Prototypes [K, d]
        self.proto = nn.Parameter(torch.randn(K, d) * (d ** -0.5))

        # 2. Intent-conditioned query projection g: [h_i ; sg(e_bar_i)] -> u_i in R^d
        self.g = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # 3. Hypergraph Message Passing Projections
        self.Wn = nn.Linear(d, d, bias=False)  # Node to Hyperedge projection
        self.We = nn.Linear(d, d, bias=False)  # Hyperedge to Node projection

        # 4. Inter-hyperedge global communication MHA
        self.edge_mha = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # 5. Normalization layers
        if norm_type == "rms" and hasattr(nn, "RMSNorm"):
            self.norm_edge = nn.RMSNorm(d)
            self.norm_node = nn.RMSNorm(d)
            self.norm_ffn = nn.RMSNorm(d)
        else:
            self.norm_edge = nn.LayerNorm(d)
            self.norm_node = nn.LayerNorm(d)
            self.norm_ffn = nn.LayerNorm(d)

        # 6. Feed-Forward Network (matching TransformerBlock in PARCO)
        if parallel_gated_kwargs is not None:
            self.ffn = ParallelGatedMLP(hidden_size=d, **parallel_gated_kwargs)
        else:
            ffn_hidden = feedforward_hidden or (4 * d)
            self.ffn = nn.Sequential(
                nn.Linear(d, ffn_hidden),
                nn.SiLU(),
                nn.Linear(ffn_hidden, d),
            )

    def forward(
        self,
        h: torch.Tensor,
        e_bar: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        detach_e_bar: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for CD-Hypergraph message passing.

        Args:
            h (torch.Tensor): Agent state features of shape [B, M, d].
            e_bar (torch.Tensor): Expected target embeddings of shape [B, M, d].
            prior (Optional[torch.Tensor]): Optional geometric prior of shape [B, M, K].
            beta (float): Weight for the geometric prior (decays to 0 after warm-up).
            mask (Optional[torch.Tensor]): Active agent mask of shape [B, M] (1=active, 0=inactive).
            detach_e_bar (bool): Whether to stop gradient on e_bar (default True).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - h_out: Refined agent embeddings of shape [B, M, d].
                - H: Soft incidence matrix of shape [B, M, K].
        """
        B, M, d = h.shape
        e_bar_in = e_bar.detach() if detach_e_bar else e_bar

        # Step 1: Compute intent-augmented representations u_i [B, M, d]
        h_intent = h.detach() if self.aux_detach_input else h
        u = self.g(torch.cat([h_intent, e_bar_in], dim=-1))

        # Step 2: Compute pairwise prototype similarity logits [B, M, K]
        logits = torch.matmul(u, self.proto.t()) / self.tau
        if prior is not None and beta > 0.0:
            logits = logits + beta * prior

        # Sigmoid per pair (i, k) -> continuous, unnormalized membership strength
        H = torch.sigmoid(logits)

        # Step 3: Top-k sparsity with unnormalized sigmoid values and optional STE
        if self.topk is not None and self.topk < self.K:
            _, top_indices = H.topk(self.topk, dim=-1)
            sparse_mask = torch.zeros_like(H).scatter_(-1, top_indices, 1.0)
            if self.topk_method == "ste":
                # Forward: continuous unnormalized sigmoid values at topk, 0 elsewhere
                # Backward: gradient flows through H for all entries via STE
                H = H * sparse_mask + (H - H.detach()) * (1.0 - sparse_mask)
            else:
                H = H * sparse_mask

        # Step 4: Mask out inactive / dead / padded agents
        if mask is not None:
            # mask shape [B, M] -> [B, M, 1]
            H = H * mask.unsqueeze(-1).to(H.dtype)

        # Step 5: HGNN Degree Normalization: H_tilde = D_v^(-1/2) * H * D_e^(-1/2)
        # dv: vertex degrees [B, M, 1]
        dv = H.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # de: hyperedge degrees [B, 1, K]
        de = H.sum(dim=1, keepdim=True).clamp_min(1e-6)
        Ht = H / (dv.sqrt() * de.sqrt())

        # Step 6: Hypergraph Message Passing
        # (a) Node -> Hyperedge Aggregation: e = Ht^T @ Wn(h) -> [B, K, d]
        h_proj = self.Wn(h)
        e = torch.matmul(Ht.transpose(1, 2), h_proj)

        # (b) Inter-Hyperedge Global Reasoning with Prototype Mask
        key_padding_mask = None
        if self.use_prototype_mask:
            de_raw = H.sum(dim=1)  # [B, K]
            key_padding_mask = (de_raw < self.eps_proto)  # True means masked out in PyTorch MHA
            # Safety: Ensure not all prototypes in a sample are masked to avoid MHA NaN
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                largest_proto = de_raw[all_masked].argmax(dim=-1, keepdim=True)
                key_padding_mask[all_masked] = key_padding_mask[all_masked].scatter(-1, largest_proto, False)

        e2, _ = self.edge_mha(e, e, e, key_padding_mask=key_padding_mask)
        e2 = self.norm_edge(e + e2)

        # (c) Hyperedge -> Node Broadcast: h_refined = h + Ht @ We(e2) -> [B, M, d]
        e_proj = self.We(e2)
        h_msg = torch.matmul(Ht, e_proj)
        h_mid = self.norm_node(h + h_msg)

        # Step 7: Transformer-style Feed-Forward Network (FFN) + Residual + Norm
        h_out = self.norm_ffn(h_mid + self.ffn(h_mid))

        return h_out, H
