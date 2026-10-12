import pytest
import torch
from parco.models.hscl_v2 import (
    CDHypergraphLayer,
    compute_co_membership,
    intent_overlap_loss,
    realized_collision_loss,
    anti_collapse_losses,
    compute_pair_auc,
    compute_hyperedge_purity,
)


def test_cd_hypergraph_forward_shapes():
    B, M, d, K = 4, 12, 64, 6
    layer = CDHypergraphLayer(d=d, K=K, num_heads=4, topk=2)

    h = torch.randn(B, M, d)
    e_bar = torch.randn(B, M, d)

    h_out, H = layer(h, e_bar)

    assert h_out.shape == (B, M, d), f"Expected h_out shape (4, 12, 64), got {h_out.shape}"
    assert H.shape == (B, M, K), f"Expected H shape (4, 12, 6), got {H.shape}"
    # Verify soft values in [0, 1]
    assert (H >= 0.0).all() and (H <= 1.0).all()
    # Check top-k sparsity: each agent should belong to at most topk hyperedges
    non_zeros = (H > 0.0).sum(dim=-1)
    assert (non_zeros <= 2).all()


def test_cd_hypergraph_masking():
    B, M, d, K = 2, 8, 32, 4
    layer = CDHypergraphLayer(d=d, K=K, num_heads=2, topk=2)

    h = torch.randn(B, M, d)
    e_bar = torch.randn(B, M, d)
    # Mask out last 3 agents
    mask = torch.ones(B, M)
    mask[:, 5:] = 0.0

    h_out, H = layer(h, e_bar, mask=mask)

    # Inactive agents must have strictly 0 incidence
    assert (H[:, 5:, :] == 0.0).all(), "Inactive agents should have zero incidence"


def test_gradient_flow_rl_loss():
    """Verify that RL reward gradient flows back through H and hypergraph message passing."""
    B, M, d, K = 2, 6, 32, 4
    layer = CDHypergraphLayer(d=d, K=K, num_heads=2, topk=2)

    h = torch.randn(B, M, d, requires_grad=True)
    e_bar = torch.randn(B, M, d)

    h_out, H = layer(h, e_bar)
    # Dummy RL loss
    loss_rl = (h_out ** 2).mean()
    loss_rl.backward()

    assert layer.proto.grad is not None, "Prototypes must receive gradient from RL loss"
    assert layer.proto.grad.norm() > 0, "Prototype gradient must be non-zero"
    assert layer.Wn.weight.grad is not None and layer.Wn.weight.grad.norm() > 0
    assert layer.We.weight.grad is not None and layer.We.weight.grad.norm() > 0
    assert h.grad is not None and h.grad.norm() > 0


def test_gradient_flow_collision_loss():
    """Verify that realized collision loss propagates back to prototype and assignment network."""
    B, M, d, K = 2, 6, 32, 4
    layer = CDHypergraphLayer(d=d, K=K, num_heads=2, topk=2)

    h = torch.randn(B, M, d)
    e_bar = torch.randn(B, M, d)
    # Synthetic actions with intentional collisions
    actions = torch.tensor([[0, 0, 1, 2, 2, 3], [1, 2, 3, 3, 4, 1]])

    h_out, H = layer(h, e_bar)
    loss_col = realized_collision_loss(H, actions, pos_weight=3.0)
    loss_col.backward()

    assert layer.proto.grad is not None and layer.proto.grad.norm() > 0
    # The assignment MLP g must receive gradients from collision supervision!
    for p in layer.g.parameters():
        assert p.grad is not None and p.grad.norm() > 0


def test_gradient_flow_intent_loss():
    """Verify that intent overlap loss propagates back to prototype and assignment network."""
    B, M, d, K, N = 2, 6, 32, 4, 20
    layer = CDHypergraphLayer(d=d, K=K, num_heads=2, topk=2)

    h = torch.randn(B, M, d)
    e_bar = torch.randn(B, M, d)
    # Tentative probabilities
    logits = torch.randn(B, M, N)
    p = torch.softmax(logits, dim=-1)

    h_out, H = layer(h, e_bar)
    loss_int = intent_overlap_loss(H, p, top_candidates=8)
    loss_int.backward()

    assert layer.proto.grad is not None and layer.proto.grad.norm() > 0
    for param in layer.g.parameters():
        assert param.grad is not None and param.grad.norm() > 0


def test_anti_collapse_losses():
    B, M, d, K = 3, 8, 32, 4
    layer = CDHypergraphLayer(d=d, K=K, num_heads=2, topk=2)

    h = torch.randn(B, M, d)
    e_bar = torch.randn(B, M, d)

    _, H = layer(h, e_bar)
    loss_bal, loss_sp = anti_collapse_losses(H)

    assert loss_bal.item() >= 0.0
    assert loss_sp.item() >= 0.0

    (loss_bal + loss_sp).backward()
    assert layer.proto.grad is not None


def test_metrics_pair_auc_and_purity():
    B, M, K = 2, 6, 3
    # Synthetic H where agents 0 and 1 belong to hyperedge 0, agents 2 and 3 belong to hyperedge 1
    H = torch.tensor([
        [[1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]],
    ])
    # Collisions exactly match hyperedge clusters!
    actions = torch.tensor([
        [5, 5, 2, 2, 9, 8],
        [4, 4, 7, 7, 1, 2],
    ])

    auc = compute_pair_auc(H, actions)
    purity = compute_hyperedge_purity(H, actions, threshold=0.5)

    assert auc > 0.8, f"Expected high AUC for aligned collisions, got {auc}"
    assert purity == 1.0, f"Expected 100% purity for aligned collisions, got {purity}"
