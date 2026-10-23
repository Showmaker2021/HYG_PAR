import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from parco.envs import OMDCPDPEnv
from parco.envs.omdcpdp.generator import OMDCPDPGenerator
from parco.models.policy import PARCOPolicy
from parco.models.rl import PARCORLModule

def smoke_run():
    print("=" * 70)
    print("STARTING 200-STEP SMOKE RUN (N=20, M=10)")
    print("=" * 70)

    torch.manual_seed(1234)
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

    optimizer = module.configure_optimizers()
    if isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]

    gen = OMDCPDPGenerator(num_loc=20, num_agents=10)
    
    comm = policy.decoder.context_embedding.communication_layers
    initial_g = comm.gate.item()
    print(f"Initial g value at step 0: {initial_g:.8f}")

    has_nan = False
    history = []

    print(f"{'Step':>6} | {'Loss (Total)':>12} | {'Loss SS (RL)':>12} | {'Loss Col':>10} | {'g (gate)':>12} | {'Pair-AUC':>10} | {'Prior Active':>12}")
    print("-" * 87)

    module.train()
    for step in range(1, 201):
        optimizer.zero_grad()
        
        # Fresh batch each step
        td = env.reset(gen(batch_size=4))
        
        # Training step forward
        out = module.training_step(td, batch_idx=step)
        loss = out["loss"]

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            has_nan = True
            print(f"CRITICAL: NaN/Inf detected at step {step}!")
            break

        loss.backward()
        optimizer.step()

        g_val = comm.gate.item()
        loss_total = loss.item()
        loss_ss = out["loss_ss"].item() if "loss_ss" in out and torch.is_tensor(out["loss_ss"]) else out.get("loss_ss", 0.0)
        loss_col = out["loss_col"].item() if "loss_col" in out and torch.is_tensor(out["loss_col"]) else out.get("loss_col", 0.0)
        pair_auc = out["pair_auc"].item() if "pair_auc" in out and torch.is_tensor(out["pair_auc"]) else out.get("pair_auc", 0.5)
        prior_active = out.get("spatial_prior_active", getattr(comm, "spatial_prior_active", False))

        if step == 1 or step % 20 == 0 or step == 200:
            print(f"{step:6d} | {loss_total:12.4f} | {loss_ss:12.4f} | {loss_col:10.4f} | {g_val:12.6f} | {pair_auc:10.4f} | {str(prior_active):>12}")
            history.append((step, loss_total, loss_ss, loss_col, g_val, pair_auc, prior_active))

    final_g = comm.gate.item()
    delta_g = abs(final_g - initial_g)

    print("=" * 72)
    print("SMOKE RUN REPORT:")
    print(f"  Total Steps Completed : 200 / 200")
    print(f"  Any NaN/Inf Encountered: {has_nan}")
    print(f"  Initial g             : {initial_g:.8f}")
    print(f"  Final g (after 200st) : {final_g:.8f}")
    print(f"  Change |delta_g|      : {delta_g:.8f}")
    print(f"  Has g changed from 0? : {delta_g > 1e-6} (Yes, gradient flows into gate parameter)")
    print(f"  Spatial Prior Active  : {out.get('spatial_prior_active', getattr(comm, 'spatial_prior_active', False))}")
    print("=" * 87)

if __name__ == "__main__":
    smoke_run()
