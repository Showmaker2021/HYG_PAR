import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def compute_co_membership(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes pairwise hyperedge co-membership similarity S_hat_ij in [0, 1].

    Args:
        H (torch.Tensor): Soft incidence matrix of shape [B, M, K].

    Returns:
        torch.Tensor: S_hat of shape [B, M, M].
    """
    H_norm = F.normalize(H, p=2, dim=-1, eps=eps)
    S_hat = torch.bmm(H_norm, H_norm.transpose(1, 2))
    return S_hat.clamp(0.0, 1.0)


def intent_overlap_loss(
    H: torch.Tensor,
    p: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    top_candidates: Optional[int] = 16,
) -> torch.Tensor:
    """Self-supervised Intent Overlap Loss (L_int).

    Computes BCE between co-membership S_hat and Bhattacharyya coefficient T_int
    derived from tentative action distributions p.

    Args:
        H (torch.Tensor): Soft incidence matrix of shape [B, M, K].
        p (torch.Tensor): Tentative action probabilities of shape [B, M, N].
        mask (Optional[torch.Tensor]): Active agent mask of shape [B, M].
        top_candidates (Optional[int]): Sparsify to top-C actions to reduce memory.

    Returns:
        torch.Tensor: Scalar BCE loss.
    """
    B, M, N = p.shape
    if top_candidates is not None and top_candidates < N:
        # Keep top-C actions per agent to avoid O(M^2 N) memory
        top_vals, top_idx = p.topk(top_candidates, dim=-1)
        p_sparse = torch.zeros_like(p).scatter_(-1, top_idx, top_vals)
        p_sparse = p_sparse / (p_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        p = p_sparse

    # Bhattacharyya coefficient: T_int_ij = sum_n sqrt(p_in * p_jn)
    p_sqrt = torch.sqrt(p.clamp_min(1e-9))
    T_int = torch.bmm(p_sqrt, p_sqrt.transpose(1, 2)).detach().clamp(0.0, 1.0)

    # Predicted co-membership S_hat
    S_hat = compute_co_membership(H)

    # Mask off-diagonal entries & inactive agents
    eye = torch.eye(M, device=H.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    valid_pair_mask = ~eye
    if mask is not None:
        agent_pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)  # [B, M, M]
        valid_pair_mask = valid_pair_mask & agent_pair_mask

    if valid_pair_mask.sum() == 0:
        return torch.tensor(0.0, device=H.device, requires_grad=True)

    s_pred = S_hat[valid_pair_mask].clamp(1e-6, 1.0 - 1e-6)
    t_target = T_int[valid_pair_mask]

    return F.binary_cross_entropy(s_pred, t_target)


def realized_collision_loss(
    H: torch.Tensor,
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Supervised Realized Collision Loss (L_col).

    Supervises S_hat with the ground-truth collisions observed when agents sample actions.

    Args:
        H (torch.Tensor): Soft incidence matrix of shape [B, M, K].
        actions (torch.Tensor): Sampled discrete actions of shape [B, M].
        mask (Optional[torch.Tensor]): Active agent mask of shape [B, M].
        pos_weight (float): Positive class weight (since collision is rare).

    Returns:
        torch.Tensor: Scalar BCE loss.
    """
    B, M = actions.shape
    # Ground truth collision matrix: T_col_ij = 1 if a_i == a_j else 0
    T_col = (actions.unsqueeze(-1) == actions.unsqueeze(1)).float()

    S_hat = compute_co_membership(H)

    eye = torch.eye(M, device=H.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    valid_pair_mask = ~eye
    if mask is not None:
        agent_pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        valid_pair_mask = valid_pair_mask & agent_pair_mask

    if valid_pair_mask.sum() == 0:
        return torch.tensor(0.0, device=H.device, requires_grad=True)

    s_pred = S_hat[valid_pair_mask].clamp(1e-6, 1.0 - 1e-6)
    t_target = T_col[valid_pair_mask]

    # Weighted BCE
    weights = torch.where(t_target == 1.0, pos_weight, 1.0)
    loss = F.binary_cross_entropy(s_pred, t_target, weight=weights)
    return loss


def anti_collapse_losses(
    H: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes load balancing loss (L_bal) and sparsity loss (L_sp) to prevent slot collapse.

    Args:
        H (torch.Tensor): Soft incidence matrix of shape [B, M, K].
        mask (Optional[torch.Tensor]): Active agent mask [B, M].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (L_bal, L_sp)
    """
    B, M, K = H.shape
    if mask is not None:
        H_valid = H * mask.unsqueeze(-1).to(H.dtype)
        valid_count = mask.sum().clamp_min(1.0)
    else:
        H_valid = H
        valid_count = float(B * M)

    # 1. Load balancing: KL(u_bar || Uniform_K)
    # Average usage per hyperedge across all agents
    u_bar = H_valid.sum(dim=(0, 1)) / valid_count  # [K]
    u_prob = (u_bar / u_bar.sum().clamp_min(1e-8)).clamp_min(1e-6)
    unif = torch.full_like(u_prob, 1.0 / K)
    loss_bal = F.kl_div(u_prob.log(), unif, reduction="sum")

    # 2. Sparsity: Mean entropy of normalized H_i
    H_norm_agent = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(H_norm_agent * torch.log(H_norm_agent.clamp_min(1e-8))).sum(dim=-1)
    if mask is not None:
        loss_sp = (entropy * mask).sum() / valid_count
    else:
        loss_sp = entropy.mean()

    return loss_bal, loss_sp


def compute_pair_auc(
    H: torch.Tensor,
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Computes Pair-AUC metric: ability of S_hat to discriminate actual collisions.

    Returns:
        float: ROC-AUC score in [0.0, 1.0].
    """
    with torch.no_grad():
        B, M = actions.shape
        T_col = (actions.unsqueeze(-1) == actions.unsqueeze(1)).float()
        S_hat = compute_co_membership(H)

        eye = torch.eye(M, device=H.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        valid_pair_mask = ~eye
        if mask is not None:
            agent_pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
            valid_pair_mask = valid_pair_mask & agent_pair_mask

        scores = S_hat[valid_pair_mask].cpu().numpy()
        targets = T_col[valid_pair_mask].cpu().numpy()

        if targets.sum() == 0 or targets.sum() == len(targets):
            return 0.5  # Neutral when no positive or negative collisions exist

        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(targets, scores))


def compute_hyperedge_purity(
    H: torch.Tensor,
    actions: torch.Tensor,
    threshold: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Computes Hyperedge Purity: proportion of colliding pairs that share at least one hyperedge.

    Returns:
        float: Purity ratio in [0.0, 1.0].
    """
    with torch.no_grad():
        B, M = actions.shape
        T_col = (actions.unsqueeze(-1) == actions.unsqueeze(1)).bool()
        eye = torch.eye(M, device=H.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        col_pairs = T_col & (~eye)

        if mask is not None:
            agent_pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
            col_pairs = col_pairs & agent_pair_mask

        total_collisions = col_pairs.sum().item()
        if total_collisions == 0:
            return 1.0

        # Two agents share a hyperedge if there exists k such that both H_ik >= threshold and H_jk >= threshold
        H_bin = H >= threshold
        # [B, M, 1, K] & [B, 1, M, K] -> [B, M, M, K] -> any along K
        shared = (H_bin.unsqueeze(2) & H_bin.unsqueeze(1)).any(dim=-1)

        captured_collisions = (shared & col_pairs).sum().item()
        return float(captured_collisions / total_collisions)
