import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from parco.envs import OMDCPDPEnv
from parco.envs.omdcpdp.generator import OMDCPDPGenerator
from parco.models.policy import PARCOPolicy
from parco.models.rl import PARCORLModule
from parco.models.hscl_v2.wrapper import CDHGCommunicationWrapper
from parco.models.hscl_v2.layer import CDHypergraphLayer
from parco.models.hscl_v2.losses import (
    intent_overlap_loss,
    realized_collision_loss,
    anti_collapse_losses,
    compute_pair_auc,
)


def test_bit_exact_when_g_is_zero():
    """Requirement A: Test that when g=0, h_comm is BIT-EXACT identical to original PARCO communication."""
    torch.manual_seed(1234)
    B, M, d = 4, 10, 128
    h = torch.randn(B, M, d)
    embeddings = torch.randn(B, 30, d)

    cfg = OmegaConf.create({
        "use_hscl": True,
        "version": "v2",
        "num_prototypes": 8,
        "tau": 1.0,
        "topk": 2,
        "use_gate": True,
    })

    flat_comm = nn.Sequential(
        nn.Linear(d, d),
        nn.RMSNorm(d),
    )

    wrapper = CDHGCommunicationWrapper(
        embed_dim=d,
        config=cfg,
        original_layers=flat_comm,
    )

    with torch.no_grad():
        wrapper.gate.zero_()

    with torch.no_grad():
        out_flat = flat_comm(h)

    with torch.no_grad():
        out_hybrid = wrapper(h, embeddings=embeddings)

    max_diff = (out_hybrid - out_flat).abs().max().item()
    print(f"\n[Test Bit-Exact] Max diff between PARCO flat and Hybrid CD-HG with g=0: {max_diff:.10e}")
    assert max_diff == 0.0, f"g=0 must be bit-exact to PARCO flat_comm, but got diff {max_diff}"


def test_aux_detach_input_gradient_isolation():
    """Requirement 2: Test that with aux_detach_input=True, aux loss grad on encoder/upstream=0;
    with aux_detach_input=False, grad is non-zero."""
    torch.manual_seed(42)
    B, M, d, K = 2, 5, 64, 8
    
    # Upstream encoder projection
    upstream_encoder = nn.Linear(d, d)
    x = torch.randn(B, M, d)
    h = upstream_encoder(x)
    embeddings = torch.randn(B, 15, d)

    # 1. Test with aux_detach_input = True
    cfg_true = OmegaConf.create({
        "use_hscl": True,
        "version": "v2",
        "num_prototypes": K,
        "tau": 1.0,
        "topk": 2,
        "use_gate": True,
        "aux_detach_input": True,
    })
    wrapper_isolated = CDHGCommunicationWrapper(embed_dim=d, config=cfg_true)
    _ = wrapper_isolated(h, embeddings=embeddings)
    H_iso = wrapper_isolated.last_H
    
    loss_aux_iso = H_iso.sum()
    upstream_encoder.zero_grad()
    loss_aux_iso.backward()
    
    grad_norm_iso = upstream_encoder.weight.grad.norm().item() if upstream_encoder.weight.grad is not None else 0.0
    print(f"\n[Test Gradient Isolation] Grad on encoder with aux_detach_input=True: {grad_norm_iso:.8f}")
    assert grad_norm_iso == 0.0, "Auxiliary loss MUST NOT leak gradient to encoder when aux_detach_input=True!"

    # 2. Test with aux_detach_input = False
    upstream_encoder.zero_grad()
    h_non_iso = upstream_encoder(x)
    cfg_false = OmegaConf.create({
        "use_hscl": True,
        "version": "v2",
        "num_prototypes": K,
        "tau": 1.0,
        "topk": 2,
        "use_gate": True,
        "aux_detach_input": False,
    })
    wrapper_leak = CDHGCommunicationWrapper(embed_dim=d, config=cfg_false)
    _ = wrapper_leak(h_non_iso, embeddings=embeddings)
    H_leak = wrapper_leak.last_H
    
    loss_aux_leak = H_leak.sum()
    loss_aux_leak.backward()
    
    grad_norm_leak = upstream_encoder.weight.grad.norm().item() if upstream_encoder.weight.grad is not None else 0.0
    print(f"[Test Gradient Isolation] Grad on encoder with aux_detach_input=False: {grad_norm_leak:.8f}")
    assert grad_norm_leak > 0.0, "Auxiliary loss MUST propagate gradient to encoder when aux_detach_input=False!"


def test_instance_adaptive_farthest_point_sampling_prior():
    """Requirement 3: Test that spatial prior C_k is dynamically adapted per instance via FPS,
    NOT fixed globally."""
    torch.manual_seed(99)
    from parco.models.hscl_v2.wrapper import farthest_point_sampling
    
    B, N, K = 2, 20, 8
    # Instance 0: located in [0.0, 0.2] corner
    locs0 = torch.rand(1, N, 2) * 0.2
    # Instance 1: located in [0.8, 1.0] corner
    locs1 = torch.rand(1, N, 2) * 0.2 + 0.8
    locs = torch.cat([locs0, locs1], dim=0)

    centroids = farthest_point_sampling(locs, K)
    print(f"\n[Test Instance FPS] Centroids 0 mean: {centroids[0].mean(dim=0).tolist()}")
    print(f"[Test Instance FPS] Centroids 1 mean: {centroids[1].mean(dim=0).tolist()}")
    
    assert (centroids[0] <= 0.25).all(), "Instance 0 centroids must adapt to instance 0 locations!"
    assert (centroids[1] >= 0.75).all(), "Instance 1 centroids must adapt to instance 1 locations!"
    assert centroids.shape == (B, K, 2)


def test_optimizer_gate_param_group_and_lr():
    """Requirement 4: Test that configure_optimizers creates a dedicated param group for gate with lr_gate."""
    env = OMDCPDPEnv()
    cfg = OmegaConf.load("configs/experiment/omdcpdp_cdhg.yaml")

    policy = PARCOPolicy(
        env_name="omdcpdp",
        agent_handler="highprob",
        embed_dim=128,
        normalization="rms",
        parallel_gated_kwargs={"mlp_activation": "silu"},
        context_embedding_kwargs=dict(cfg.model.policy.context_embedding_kwargs),
    )

    module = PARCORLModule(
        env=env,
        policy=policy,
        optimizer_kwargs={"lr": 1e-4},
        lr_gate=1e-3,
    )

    optimizer = module.configure_optimizers()
    if isinstance(optimizer, (tuple, list)):
        optimizer = optimizer[0]

    param_groups = optimizer.param_groups
    print(f"\n[Test Optimizer Groups] Number of param groups: {len(param_groups)}")
    print(f"[Test Optimizer Groups] Group 0 lr (base): {param_groups[0]['lr']}")
    print(f"[Test Optimizer Groups] Group 1 lr (gate): {param_groups[1]['lr']}")
    
    assert len(param_groups) == 2, "Optimizer must contain 2 param groups (other and gate)!"
    assert param_groups[0]["lr"] == 1e-4, "Base param group must have lr=1e-4"
    assert param_groups[1]["lr"] == 1e-3, "Gate param group must have lr_gate=1e-3 (10x base)"


def test_loss_gradients_on_H():
    """Requirement C: Test that each auxiliary loss has non-zero gradients on H."""
    B, M, K = 4, 10, 8
    N = 25
    
    # 1. Intent Overlap Loss (L_int)
    H1 = torch.rand(B, M, K, requires_grad=True)
    p = F.softmax(torch.randn(B, M, N), dim=-1)
    l_int = intent_overlap_loss(H1, p)
    l_int.backward()
    grad1_norm = H1.grad.norm().item()
    print(f"\n[Test Gradients] L_int grad norm on H: {grad1_norm:.6f}")
    assert grad1_norm > 0, "L_int must have non-zero gradient on H!"

    # 2. Realized Collision Loss (L_col)
    H2 = torch.rand(B, M, K, requires_grad=True)
    actions = torch.tensor([
        [0, 1, 2, 0, 1, 3, 4, 5, 2, 6],
        [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 4],
        [5, 4, 3, 2, 1, 0, 5, 4, 3, 2],
    ])
    l_col = realized_collision_loss(H2, actions)
    l_col.backward()
    grad2_norm = H2.grad.norm().item()
    print(f"[Test Gradients] L_col grad norm on H: {grad2_norm:.6f}")
    assert grad2_norm > 0, "L_col must have non-zero gradient on H!"

    # 3. Load Balancing Loss (L_bal)
    H3 = torch.rand(B, M, K, requires_grad=True)
    l_bal, l_sp = anti_collapse_losses(H3)
    l_bal.backward()
    grad3_norm = H3.grad.norm().item()
    print(f"[Test Gradients] L_bal grad norm on H: {grad3_norm:.6f}")
    assert grad3_norm > 0, "L_bal must have non-zero gradient on H!"

    # 4. Sparsity Loss (L_sp)
    H4 = torch.rand(B, M, K, requires_grad=True)
    _, l_sp2 = anti_collapse_losses(H4)
    l_sp2.backward()
    grad4_norm = H4.grad.norm().item()
    print(f"[Test Gradients] L_sp grad norm on H: {grad4_norm:.6f}")
    assert grad4_norm > 0, "L_sp must have non-zero gradient on H!"


def test_ste_topk_and_continuous_H():
    """Requirement B: Test that H retains continuous unnormalized sigmoid values and STE passes gradients."""
    d, K, topk = 64, 8, 2
    layer = CDHypergraphLayer(d=d, K=K, topk=topk, topk_method="ste")
    
    h = torch.randn(2, 6, d, requires_grad=True)
    e_bar = torch.randn(2, 6, d)
    
    h_out, H = layer(h, e_bar)
    
    non_zeros = H[H > 0]
    unique_vals = len(torch.unique(non_zeros.round(decimals=4)))
    print(f"\n[Test Continuous H] Number of unique non-zero values in H: {unique_vals}")
    assert unique_vals > 2, "H must retain continuous unnormalized sigmoid values!"

    loss = h_out.mean() + H.mean()
    loss.backward()
    assert layer.proto.grad is not None and layer.proto.grad.norm().item() > 0, "STE must pass gradients to proto!"
    assert h.grad is not None and h.grad.norm().item() > 0, "STE must pass gradients to h!"
    print(f"[Test STE Gradient] Proto grad norm: {layer.proto.grad.norm().item():.6f}")
    print(f"[Test STE Gradient] h grad norm: {h.grad.norm().item():.6f}")


def test_prototype_mask_prevents_nan():
    """Requirement B: Test that empty hyperedges (degree < eps) are masked in MHA without causing NaN."""
    d, K, topk = 64, 8, 2
    layer = CDHypergraphLayer(d=d, K=K, topk=topk, use_prototype_mask=True, eps_proto=1e-3)
    
    # 2 agents choosing only 2 out of 8 hyperedges -> 6 hyperedges have degree == 0
    h = torch.randn(2, 2, d)
    e_bar = torch.randn(2, 2, d)
    
    h_out, H = layer(h, e_bar)
    assert not torch.isnan(h_out).any(), "Prototype mask caused NaN in h_out!"
    assert not torch.isinf(h_out).any(), "Prototype mask caused Inf in h_out!"
    print(f"\n[Test Prototype Mask] Successfully executed with empty hyperedges. NaN check: {torch.isnan(h_out).any().item()}")


def test_full_policy_end_to_end_step():
    """Test full policy and PARCORLModule training step with all metrics logged."""
    env = OMDCPDPEnv()
    cfg = OmegaConf.load("configs/experiment/omdcpdp_cdhg.yaml")

    policy = PARCOPolicy(
        env_name="omdcpdp",
        agent_handler="highprob",
        embed_dim=128,
        normalization="rms",
        parallel_gated_kwargs={"mlp_activation": "silu"},
        context_embedding_kwargs=dict(cfg.model.policy.context_embedding_kwargs),
    )

    module = PARCORLModule(
        env=env,
        policy=policy,
        lambda_int=cfg.model.lambda_int,
        lambda_col=cfg.model.lambda_col,
        lambda_bal=cfg.model.lambda_bal,
        lambda_sp=cfg.model.lambda_sp,
        warmup_ratio=cfg.model.warmup_ratio,
        lr_gate=cfg.model.lr_gate,
    )

    gen = OMDCPDPGenerator(num_loc=50, num_agents=10)
    td = env.reset(gen(batch_size=2))

    module.train()
    out = module.training_step(td, batch_idx=0)
    
    print("\n[Test End-to-End Metrics]")
    print(f"  loss: {out['loss'].item():.4f}")
    for k in ["loss_col", "train/loss_col", "gate_val", "train/gate_val", "pair_auc", "train/pair_auc"]:
        if k in out:
            val = out[k]
            v = val.item() if torch.is_tensor(val) else val
            print(f"  {k}: {v:.6f}")
    
    assert not torch.isnan(out["loss"]), "Training step loss is NaN!"
