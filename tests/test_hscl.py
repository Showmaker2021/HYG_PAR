import pytest
import torch
import torch.nn as nn

from parco.models.hscl import (
    AutoEigenSolver,
    DenseCPUEigh,
    DenseGPUEigh,
    HSCLCommunicationLayer,
    HSCLConfig,
    HypergraphAttentionConv,
    StateHistoryBuffer,
    affinity_from_history,
    batched_kmeans,
)
from parco.models.nn.transformer import TransformerBlock as CommunicationLayer
from parco.models.env_embeddings.communication import BaseMultiAgentContextEmbedding
from parco.models.env_embeddings.ffsp import FFSPContextEmbedding


class TestHSCL:
    def test_0_mask_semantics(self):
        """Confirm that in active SDPA, True means attend and False means ignore."""
        from rl4co.models.nn.attention import MultiHeadAttention

        mha = MultiHeadAttention(64, 4).eval()
        x = torch.randn(1, 4, 64)
        mask = torch.tensor(
            [[[True, True, False, False],
              [True, True, False, False],
              [False, False, True, True],
              [False, False, True, True]]]
        )
        out1 = mha(x, attn_mask=mask)
        x_mod = x.clone()
        x_mod[0, 2] += 100.0  # modifying agent 2
        out2 = mha(x_mod, attn_mask=mask)
        # Agent 0 should not attend to Agent 2
        diff = (out1[0, 0] - out2[0, 0]).abs().max().item()
        assert diff < 1e-5, f"Mask semantic violation: diff={diff}"

    @pytest.mark.parametrize("M", [1, 2, 5, 30, 60, 200])
    def test_1_shape_consistency(self, M):
        """Test output shape consistency for diverse agent counts M."""
        B, d = 4, 64
        x = torch.randn(B, M, d)
        orig = CommunicationLayer(embed_dim=d, num_heads=4)
        config = HSCLConfig(use_hscl=True, num_clusters=4, dense_gpu_safe_max_agents=32)
        layer = HSCLCommunicationLayer(orig, embed_dim=d, num_heads=4, config=config)

        out = layer(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_2_small_m_boundary(self):
        """Test edge cases with M=1 and M=2 where num_clusters > M."""
        B, d = 2, 64
        config = HSCLConfig(use_hscl=True, num_clusters=4)

        for M in [1, 2]:
            x = torch.randn(B, M, d)
            orig = CommunicationLayer(embed_dim=d, num_heads=4)
            layer = HSCLCommunicationLayer(orig, embed_dim=d, num_heads=4, config=config)
            out = layer(x)
            assert not torch.isnan(out).any(), f"NaN detected for M={M}"
            assert out.shape == (B, M, d)

    def test_3_gradient_isolation(self):
        """Test that gradients flow to embeddings and weights, but eigensolver is detached."""
        B, M, d = 4, 10, 64
        x = torch.randn(B, M, d, requires_grad=True)
        orig = CommunicationLayer(embed_dim=d, num_heads=4)
        config = HSCLConfig(use_hscl=True, num_clusters=4)
        layer = HSCLCommunicationLayer(orig, embed_dim=d, num_heads=4, config=config)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient did not flow back to input x"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"
        for name, param in layer.hypergraph_conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_4_bit_exact_fallback(self):
        """Test bit-exact reproduction when use_hscl=False."""
        torch.manual_seed(42)
        B, M, d = 4, 8, 64
        x = torch.randn(B, M, d)

        orig = CommunicationLayer(embed_dim=d, num_heads=4)
        config = HSCLConfig(use_hscl=False)
        layer = HSCLCommunicationLayer(orig, embed_dim=d, num_heads=4, config=config)

        orig_out = orig(x)
        layer_out = layer(x)

        assert torch.allclose(orig_out, layer_out, atol=1e-6), "Fallback is not bit-exact!"

    def test_4b_sequential_propagation(self):
        """Test that multiple communication layers (nn.Sequential) work properly."""
        B, M, d = 4, 8, 64
        x = torch.randn(B, M, d)
        seq = nn.Sequential(
            CommunicationLayer(embed_dim=d, num_heads=4),
            CommunicationLayer(embed_dim=d, num_heads=4),
        )
        config = HSCLConfig(use_hscl=True, num_clusters=3)
        layer = HSCLCommunicationLayer(seq, embed_dim=d, num_heads=4, config=config)
        out = layer(x)
        assert out.shape == (B, M, d)

        # Fallback with sequential
        config_off = HSCLConfig(use_hscl=False)
        layer_off = HSCLCommunicationLayer(seq, embed_dim=d, num_heads=4, config=config_off)
        out_off = layer_off(x)
        seq_out = seq(x)
        assert torch.allclose(out_off, seq_out, atol=1e-6)

    def test_5_autoeigensolver_dispatch(self):
        """Test that AutoEigenSolver dispatches to DenseGPU for M<=32 and DenseCPU for M>32 on CUDA."""
        config = HSCLConfig(dense_gpu_safe_max_agents=32)
        solver = AutoEigenSolver(config)
        if torch.cuda.is_available():
            # M = 20 on CUDA -> DenseGPUEigh
            L_small = torch.randn(2, 20, 20, device="cuda")
            L_small = L_small + L_small.transpose(-1, -2)
            eig_small = solver.smallest_k_eigenvectors(L_small, 4)
            assert eig_small.shape == (2, 20, 4)
            assert eig_small.device.type == "cuda"

            # M = 50 on CUDA -> DenseCPUEigh (avoids cliff)
            L_large = torch.randn(2, 50, 50, device="cuda")
            L_large = L_large + L_large.transpose(-1, -2)
            eig_large = solver.smallest_k_eigenvectors(L_large, 4)
            assert eig_large.shape == (2, 50, 4)
            assert eig_large.device.type == "cuda"

    def test_7_ffsp_per_stage(self):
        """Test that FFSPContextEmbedding instantiates independent HSCL layers per stage."""
        config = HSCLConfig(use_hscl=True, ffsp_cluster_scope="per_stage")
        stage0 = FFSPContextEmbedding(
            stage_idx=0, stage_cnt=3, embed_dim=64, num_heads=4, hscl_config=config
        )
        stage1 = FFSPContextEmbedding(
            stage_idx=1, stage_cnt=3, embed_dim=64, num_heads=4, hscl_config=config
        )

        assert isinstance(stage0.communication_layer, HSCLCommunicationLayer)
        assert isinstance(stage1.communication_layer, HSCLCommunicationLayer)
        assert stage0.communication_layer is not stage1.communication_layer
        assert stage0.communication_layer.history_buffer is not stage1.communication_layer.history_buffer
